import os
import torch
import torch.onnx
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
import argparse
import time
import torchao
import sys
import os
sys.path.insert(0, os.path.expanduser('~/BaseNet'))
from basenet.models.lm_fine_tune import Fine_Tune_Model
from basenet.utils.decoder import GreedyDecoder
from fast_ctc_decode import beam_search
from collections import OrderedDict
import Levenshtein as Lev
import quanto
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import get_cosine_schedule_with_warmup
from quantized_wav2vec2 import QuantizedAttention, quantize_wav2vec_attention
from evaluate import IsolatedQuantizationEvaluator
import copy

class MyDataSet(torch.utils.data.Dataset):
  def __init__(self, signal_path, label_path, label_length_path):
    super(MyDataSet, self).__init__()
    # store the raw tensors
    self.signal = np.load(signal_path)
    self.label = np.load(label_path)
    self.label_length_path = np.load(label_length_path)

  def __len__(self):
    # a DataSet must know it size
    return self.signal.shape[0]

  def __getitem__(self, index):
    signal = self.signal[index]
    label = self.label[index]
    label_length = self.label_length_path[index]
    return signal, label, label_length


class TrainBatchBasecallDataset(Data.Dataset):
    def __init__(self, signal_path, label_path, label_length_path):
        file_list = os.listdir(signal_path)
        self.file_count = len(file_list)
        print('file number:', self.file_count)
        self.file_list = file_list
        self.signal_path = signal_path
        self.label_path = label_path
        self.label_length_path = label_length_path
        self.signals = list()
        self.labels = list()
        self.idx = 0

    def __len__(self):
        return self.file_count

    def __getitem__(self, item):
        signal = np.load(self.signal_path+'/'+self.file_list[item])
        label = np.load(self.label_path+'/'+self.file_list[item])
        label_length = np.load(self.label_length_path+'/'+self.file_list[item])
        return signal, label, label_length
    
class TrainBatchProvider():
    def __init__(self, dataset, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.dataiter = None
        self.sample_number = 0
        self.signal_pool, self.label_pool, self.label_length_pool = [], [], [] 

    def build(self):
        dataloader = Data.DataLoader(
            self.dataset, batch_size=1, shuffle=self.shuffle) #all fast5 files
        self.dataiter = dataloader.__iter__()
 
    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            while self.sample_number < self.batch_size:
                signal, label, label_length = next(self.dataiter)
                #torch.Size([1, 47, 2048]) torch.Size([1, 47]) torch.Size([1, 47, 512]) torch.Size([1, 47])
                signal = torch.squeeze(signal, dim=0)
                label = torch.squeeze(label, dim=0)
                label_length = torch.squeeze(label_length,dim=0)
                self.sample_number += signal.shape[0]
                self.signal_pool.append(signal)
                self.label_pool.append(label)
                self.label_length_pool.append(label_length)

            whole_signal = torch.cat(self.signal_pool, dim=0)
            whole_label = torch.cat(self.label_pool, dim=0)
            whole_label_length = torch.cat(self.label_length_pool, dim=0)

            batch_signal = whole_signal[:self.batch_size]
            batch_label = whole_label[:self.batch_size]
            batch_label_length = whole_label_length[:self.batch_size]

            self.signal_pool = [whole_signal[self.batch_size:]]
            self.label_pool = [whole_label[self.batch_size:]]
            self.label_length_pool = [whole_label_length[self.batch_size:]]

            self.sample_number -= self.batch_size
            #torch.Size([32, 2048]) torch.Size([32]) torch.Size([32, 512]) torch.Size([32])
            #print(batch_signal.shape, batch_signal_length.shape, batch_label.shape, batch_label_length.shape)
            return torch.unsqueeze(batch_signal, dim=2).permute(0,2,1), batch_label, batch_label_length
        except StopIteration:
            return None, None, None
        
def create_active_heads(
    num_hidden_layers: int,
    num_heads: int,
    pruned_heads_per_layer: list[list[int]],
    device: torch.device = torch.device("cpu")
) -> list[torch.Tensor]:

    active_heads_list = []
    for i in range(num_hidden_layers):
        # Start with all heads active = True
        mask = torch.ones(num_heads, dtype=torch.bool, device=device)
        # Set pruned heads to False
        for head_idx in pruned_heads_per_layer[i]:
            if 0 <= head_idx < num_heads:
                mask[head_idx] = False
            else:
                raise ValueError(f"Head index {head_idx} is out of range [0, {num_heads-1}] for layer {i}.")
        active_heads_list.append(mask)

    return active_heads_list

def create_active_heads_id(
    num_hidden_layers: int,
    num_heads: int,
    pruned_heads_per_layer: list[list[int]],
    device: torch.device = torch.device("cpu")
) -> list[torch.Tensor]:
    active_heads_id_list=[]
    for i in range(num_hidden_layers):
        active_heads_id = []
        for head_id in range(num_heads):
            if head_id not in pruned_heads_per_layer[i]:
                active_heads_id.append(head_id)
        active_heads_id_list.append( torch.tensor(active_heads_id, dtype=torch.int32, device = device))
    return active_heads_id_list

def zero_out_pruned_heads_state_dict_fine_tune(state_dict, pruned_heads_per_layer, encoder_layer_name, embed_dim, num_heads):
    head_dim = embed_dim // num_heads  # Dimension per head
    # The order here is assumed to be "k_proj", "v_proj", "q_proj"
    projections = ["k_proj", "v_proj", "q_proj"]

    # Process each layer.
    for layer_idx, pruned_heads in enumerate(pruned_heads_per_layer):
        layer_prefix = f"{encoder_layer_name}.{layer_idx}"
        
        for proj_name in projections:
            # --- Process the projection weight ---
            key_weight = f"{layer_prefix}.attention.{proj_name}.weight"
            if key_weight in state_dict:
                weight = state_dict[key_weight]
                new_weight = weight.clone()  # Clone to avoid in-place issues
                # Assume weight shape is (embed_dim, embed_dim).
                # Reshape to (num_heads, head_dim, embed_dim) to access each head separately.
                new_weight = new_weight.view(num_heads, head_dim, embed_dim)
                # Zero out the slices corresponding to pruned heads.
                for head in pruned_heads:
                    if head < num_heads:
                        new_weight[head, :, :] = 0
                    else:
                        raise ValueError(
                            f"Layer {layer_idx}: Head index {head} is out of range (num_heads={num_heads})."
                        )
                # Reshape back to (embed_dim, embed_dim)
                new_weight = new_weight.view(embed_dim, embed_dim)
                state_dict[key_weight] = new_weight

            # --- Process the projection bias (if present) ---
            key_bias = f"{layer_prefix}.attention.{proj_name}.bias"
            if key_bias in state_dict:
                bias = state_dict[key_bias]
                new_bias = bias.clone()
                # Assume bias shape is (embed_dim,)
                new_bias = new_bias.view(num_heads, head_dim)
                for head in pruned_heads:
                    if head < num_heads:
                        new_bias[head, :] = 0
                    else:
                        raise ValueError(
                            f"Layer {layer_idx}: Head index {head} is out of range (num_heads={num_heads})."
                        )
                new_bias = new_bias.view(-1)
                state_dict[key_bias] = new_bias

            # --- Process out_proj.weight ---
            key_out_proj_weight = f"{layer_prefix}.attention.out_proj.weight"
            if key_out_proj_weight in state_dict:
                # out_proj.weight is expected to be of shape (embed_dim, embed_dim)
                out_proj_weight = state_dict[key_out_proj_weight]
                new_out_proj_weight = out_proj_weight.clone()
                # Reshape to (embed_dim, num_heads, head_dim)
                new_out_proj_weight = new_out_proj_weight.view(embed_dim, num_heads, head_dim)
                for head in pruned_heads:
                    if head < num_heads:
                        new_out_proj_weight[:, head, :] = 0
                    else:
                        raise ValueError(f"Layer {layer_idx}: Head index {head} is out of range (num_heads={num_heads}).")
                # Reshape back to the original shape and update the state dict.
                state_dict[key_out_proj_weight] = new_out_proj_weight.view(embed_dim, embed_dim)
            

        
    return state_dict

beamsize=5
threshold=1e-3
alphabet = [ "N", "A", "C", "G", "T" ]
chunk_path = '/home/fengyilin/bonito/bonito/data/dna_r9.4.1/chunks.npy'
length_path = '/home/fengyilin/bonito/bonito/data/dna_r9.4.1/reference_lengths.npy'
reference_path = '/home/fengyilin/bonito/bonito/data/dna_r9.4.1/references.npy'
valid_signal_path="/home/fengyilin/basecaller_data/validation/signal_out"
valid_label_path = "/home/fengyilin/basecaller_data/validation/label_out"
valid_label_length_path="/home/fengyilin/basecaller_data/validation/label_length_out"
device =  torch.device("cuda")
batch_size = 64
epoch = 1

d_model= 768
num_heads = 12
num_hidden_layers = 12
head_dim = d_model//num_heads
pruned_heads_per_layer = [[0,  1,  2,  4,  6,  7,  9, 10, 11], [2,  5,  8, 10], [1, 3, 5, 6], [6, 7, 8], [8], 
                          [3], [6, 11], [], [5, 10, 11], [0, 1, 4], [5], []]

pruned_heads_per_layer=[[0, 1, 2, 4, 6, 7, 9, 10, 11], [1, 2, 3, 5, 8, 9, 10], [1, 3, 5, 6, 11], [2, 4, 6, 7, 8], [ 4, 8],
                         [3], [6,  11], [], [5,  10, 11], [0, 1, 4,], [1, 4, 5, 7], [1, 2, 4]]

active_heads_list = create_active_heads(num_hidden_layers,num_heads,pruned_heads_per_layer,device)
active_heads_id_list = create_active_heads_id(num_hidden_layers,num_heads,pruned_heads_per_layer,device)
fine_model = Fine_Tune_Model().to(device)
#ckpt = torch.load(r'/home/fengyilin/BaseNet/Fine_tuned_ckpt.pt')
#new_state_dict = OrderedDict({k[7:]:v for k, v in ckpt.items()})

'''
for i in range(0,12):
    old_weights = new_state_dict['sub_model.encoder.layers.'+str(i)+'.feed_forward.intermediate_dense.weight']
    new_shape = (2304, 768)  # New shape
    adjusted_weights = old_weights[ :new_shape[0],:]  # Truncate to match new shape
    new_state_dict['sub_model.encoder.layers.'+str(i)+'.feed_forward.intermediate_dense.weight'] = adjusted_weights

    old_bais = new_state_dict['sub_model.encoder.layers.'+str(i)+'.feed_forward.intermediate_dense.bias']
    adjusted_bais = old_bais[:2304]
    new_state_dict['sub_model.encoder.layers.'+str(i)+'.feed_forward.intermediate_dense.bias'] = adjusted_bais

    old_weights = new_state_dict['sub_model.encoder.layers.'+str(i)+'.feed_forward.output_dense.weight']
    new_shape = (768,2304)
    adjusted_weights = old_weights[ : ,:new_shape[1]]  # Truncate to match new shape
    new_state_dict['sub_model.encoder.layers.'+str(i)+'.feed_forward.output_dense.weight'] = adjusted_weights
'''

#sub_model.encoder.layers.0.attention.k_proj.bias', 'sub_model.encoder.layers.0.attention.v_proj.weight',
#'sub_model.encoder.layers.0.attention.v_proj.bias', 'sub_model.encoder.layers.0.attention.q_proj.weight', 
#'sub_model.encoder.layers.0.attention.q_proj.bias', 'sub_model.encoder.layers.0.attention.out_proj.weight', 
#'sub_model.encoder.layers.0.attention.out_proj.bias',


state_dict = torch.load("/home/fengyilin/BaseNet/best_fine_tuned_ctc.pt",map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#modified_state_dict = zero_out_pruned_heads_state_dict_fine_tune(state_dict, pruned_heads_per_layer, "sub_model.encoder.layers" ,d_model, num_heads)
fine_model.load_state_dict(state_dict, strict= False)
fine_model.to(dtype=torch.float16)
fine_model.eval()
original_model = copy.deepcopy(fine_model)
quantized_model = quantize_wav2vec_attention(
    fine_model,
    layers_to_quantize=None  # Quantize all layers
)



#print(summary(model,torch.zeros((1, 1, 2048),device = torch.device('cuda:0')))) 
#torch._dynamo.config.suppress_errors = True
dataset = MyDataSet(chunk_path,reference_path,length_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
valid_dataset = TrainBatchBasecallDataset(signal_path=valid_signal_path,  label_path=valid_label_path, label_length_path=valid_label_length_path)
valid_provider = TrainBatchProvider(valid_dataset, batch_size, shuffle=False)
#summary(fine_model.sub_model.encoder.layers[0].layer_norm, input_size=(1,717,768))

def simple_quantize_test(layer_idx):
    orig = original_model.sub_model.encoder.layers[layer_idx].attention.q_proj
    
    # Simple INT8 quantization without fancy scaling
    scale = orig.weight.abs().max() / 127
    quantized = torch.round(orig.weight / scale).clamp(-128, 127)
    dequantized = quantized * scale
    
    # Test
    x = torch.randn(1, 128, orig.in_features).half().to(orig.weight.device)
    orig_out = F.linear(x, orig.weight)
    quant_out = F.linear(x, dequantized)
    
    mse = ((orig_out - quant_out)**2).mean()
    cosine = F.cosine_similarity(orig_out.flatten(), quant_out.flatten(), dim=0)
    
    print(f"Simple quant - MSE: {mse:.2e}, Cosine: {cosine:.4f}")

for i in range(12):
    print(f"Layer {i} attention quantization test:")
    simple_quantize_test(i)



evaluator = IsolatedQuantizationEvaluator(original_model, quantized_model)
for i, (signal, label, label_length) in enumerate(data_loader):
        print(i)
        #signal=torch.unsqueeze(signal,dim=2).permute(0,2,1)#joint_model
        if i>1:
           break
       
        signal= signal.to(dtype=torch.float16,device=device) # (N,L,C), [32,2048,1]
        #logits = model(signal).transpose(1,0).to(torch.float32) # (N,L,C)
        metrics_df = evaluator.evaluate_all_isolated(signal)
        # 1. Basic Summary Statistics
        print("\n=== QUANTIZATION EVALUATION SUMMARY ===")
        print(f"\nOverall Statistics:")
        print(f"  Average MSE: {metrics_df['mse'].mean():.2e}")
        print(f"  Average Cosine Similarity: {metrics_df['cosine_similarity'].mean():.4f}")
        print(f"  Worst MSE: {metrics_df['mse'].max():.2e}")
        print(f"  Best Cosine Similarity: {metrics_df['cosine_similarity'].max():.4f}")
        print(f"  Worst Cosine Similarity: {metrics_df['cosine_similarity'].min():.4f}")

        # 2. Per Module Type Analysis
        print("\n=== PER MODULE TYPE ===")
        module_stats = metrics_df.groupby('module').agg({
            'mse': ['mean', 'std', 'max'],
            'cosine_similarity': ['mean', 'std', 'min']
        }).round(4)
        print(module_stats)

        # 3. Per Layer Analysis
        print("\n=== PER LAYER ===")
        layer_stats = metrics_df.groupby('layer').agg({
            'mse': 'mean',
            'cosine_similarity': 'mean'
        }).round(4)
        print(layer_stats)


#test
'''
list_charcter_error = []
start = time.time()
target_decoder = GreedyDecoder('NACGT ', blank_index=0)

start = time.time()
total_loss = []


with torch.no_grad():
    total_wer, total_cer, num_tokens, num_chars = 0, 0, 0, 0
    for i, (signal, label, label_length) in enumerate(data_loader):
        #signal=torch.unsqueeze(signal,dim=2).permute(0,2,1)#joint_model
        if i>1:
           break
        print(i) # torch.Size([32, 3600]) torch.Size([32, 480]) torch.Size([32])
        signal= signal.to(dtype=torch.float32,device=device) # (N,L,C), [32,2048,1]
        #logits = model(signal).transpose(1,0).to(torch.float32) # (N,L,C)
        logits, attn_score_list = fine_model(signal)
        for layer_idx in range(12):
            values = attn_score_list[layer_idx][0].cpu()
            hist = torch.histc(values, bins=100, min=1e-8, max=1e-2)
            edges = torch.logspace(-8, -2, steps=101)
            plt.bar(edges[:-1].cpu().numpy(), hist.cpu().numpy(), width=np.diff(edges.cpu().numpy()), align='edge')
            plt.xscale('log')
            plt.title(f"Layer {layer_idx} Attention (1e-8 to 1e-2)")
            plt.savefig(f'./attention_plots/attention_layer_{layer_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()  # Explicitly close the figure
            print(f"Saved layer {layer_idx} visualization")
            
        
        logits = logits.to(torch.float32)
        sequence_strings = []
        for logit in torch.exp(logits).cpu().detach().numpy():
            sequence, path = beam_search(logit, alphabet, beamsize, threshold)
            sequence_strings.append(sequence)
            #print('>>',sequence)
        reference_strings = target_decoder.convert_to_strings(label, label_length)
        for x in range(len(label)):
            sequence, reference = sequence_strings[x], reference_strings[x][0]
            sequence, reference, = sequence.replace(' ', ''), reference.replace(' ', '')
            cer_inst = Lev.distance(sequence, reference)
            total_cer += cer_inst
            num_chars += len(reference)
 
    cer = float(total_cer) / num_chars
    list_charcter_error.append(cer)
    print('charcter error {cer:.3f}% time: {time:.3f} s'.format(
            cer=cer * 100,
            time=time.time() - start))
    

'''