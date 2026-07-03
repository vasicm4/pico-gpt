import numpy as np
from layers.normalization import RMSNorm
from layers.activation import SwiGLU
from layers.position import RoPE
from layers.cqa import CausalGQABlock
from layers.oracle import PicoGPTOracle

print('Testing RMSNorm')
rms = RMSNorm(16)
x = np.random.randn(2,5,16).astype(np.float32)
out = rms.forward(x)
print('Output shape:', out.shape)

print('Testing SwiGLU')
swi = SwiGLU(d_model=16, d_ffn=24)
out2 = swi.forward(x)
print('Output shape:', out2.shape)

print('Testing RoPE')
rope = RoPE(head_dim=4, max_seq_len=10)
x3 = np.random.randn(2,5,4).astype(np.float32)
out3 = rope.forward(x3)
print('Output shape:', out3.shape)

print('Testing CausalGQABlock')
cqa = CausalGQABlock(d_model=16, n_heads=4, head_dim=4, max_seq_len=10)
x4 = np.random.randn(2,5,16).astype(np.float32)
out4 = cqa.forward(x4)
print('Output shape:', out4.shape)

print('Testing PicoGPTOracle')
model = PicoGPTOracle(vocab_size=10, d_model=16, n_layers=2, n_heads=4, max_seq_len=10)
idx = np.random.randint(0,10,size=(2,5))
logits, loss = model.forward(idx)
print('Logits shape:', logits.shape)
print('Loss:', loss)
print('All tests passed')