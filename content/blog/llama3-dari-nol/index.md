---
title: 📝 Penjelasan dan Implementasi llama3 dari Nol
summary: Dalam artikel ini, saya mengimplementasikan Llama3 dari nol, satu tensor dan perkalian matriks pada satu waktu.
type: post
date: 2024-05-19T13:20:01+00:00
authors:
  - fdciabdul
tags:
  - DevOps
  - Linux
---

# Penjelasan dan Implementasi llama3 dari Nol
Dalam artikel ini, saya mengimplementasikan Llama3 dari nol, satu tensor dan perkalian matriks pada satu waktu.
<br>
Selain itu, saya akan memuat tensor langsung dari file model yang disediakan oleh Meta untuk Llama3. Anda perlu mengunduh bobot sebelum menjalankan file ini.
Berikut adalah tautan resmi untuk mengunduh bobot: [https://llama.meta.com/llama-downloads/](https://llama.meta.com/llama-downloads/)

<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/archi.png"/>
</div>

## Tokenizer
Saya tidak akan mengimplementasikan tokenizer BPE (tapi Andrej Karpathy memiliki implementasi yang sangat bersih)
<br>
Tautan ke implementasinya: [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)

<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/karpathyminbpe.png" width="600"/>
</div>

```python
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt

tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
special_tokens = [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

tokenizer.decode(tokenizer.encode("hello world!"))
```

    'hello world!'

## Membaca File Model
Biasanya, membaca ini tergantung pada bagaimana kelas model ditulis dan nama variabel di dalamnya.
<br>
Namun, karena kita mengimplementasikan Llama3 dari nol, kita akan membaca file satu tensor pada satu waktu.
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/model.png" width="600"/>
</div>

```python
model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
print(json.dumps(list(model.keys())[:20], indent=4))
```

    [
        "tok_embeddings.weight",
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w3.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.1.attention.wq.weight",
        "layers.1.attention.wk.weight",
        "layers.1.attention.wv.weight",
        "layers.1.attention.wo.weight",
        "layers.1.feed_forward.w1.weight",
        "layers.1.feed_forward.w3.weight",
        "layers.1.feed_forward.w2.weight",
        "layers.1.attention_norm.weight",
        "layers.1.ffn_norm.weight",
        "layers.2.attention.wq.weight"
    ]

```python
with open("Meta-Llama-3-8B/params.json", "r") as f:
    config = json.load(f)
config
```

    {'dim': 4096,
     'n_layers': 32,
     'n_heads': 32,
     'n_kv_heads': 8,
     'vocab_size': 128256,
     'multiple_of': 1024,
     'ffn_dim_multiplier': 1.3,
     'norm_eps': 1e-05,
     'rope_theta': 500000.0}

## Menggunakan Konfigurasi untuk Mengetahui Detail Model
1. Model memiliki 32 lapisan transformer.
2. Setiap blok perhatian multi-kepala memiliki 32 kepala.
3. Ukuran kosakata, dan sebagainya.

```python
dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])
```

## Mengonversi Teks ke Token
Di sini kita menggunakan tiktoken (saya pikir ini adalah perpustakaan OpenAI) sebagai tokenizer.
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/tokens.png" width="600"/>
</div>

```python
prompt = "the answer to the ultimate question of life, the universe, and everything is "
tokens = [128000] + tokenizer.encode(prompt)
print(tokens)
tokens = torch.tensor(tokens)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
print(prompt_split_as_tokens)
```

    [128000, 1820, 4320, 311, 279, 17139, 3488, 315, 2324, 11, 279, 15861, 11, 323, 4395, 374, 220]
    ['', 'the', ' answer', ' to', ' the', ' ultimate', ' question', ' of', ' life', ',', ' the', ' universe', ',', ' and', ' everything', ' is', ' ']

## Mengonversi Token menjadi Embedding
MAAF tapi ini adalah satu-satunya bagian dari kode di mana saya menggunakan modul jaringan saraf bawaan.
<br>
Jadi, token [17x1] kita sekarang menjadi [17x4096], yaitu 17 embedding (satu untuk setiap token) dengan panjang 4096.
<br>
<br>
Catatan: perhatikan bentuknya, ini membuatnya jauh lebih mudah untuk dipahami.

<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/embeddings.png" width="600"/>
</div>

```python
embedding_layer = torch.nn.Embedding(vocab_size, dim)
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
token_embeddings_unnormalized.shape
```

    torch.Size([17, 4096])

## Normalisasi Embedding Menggunakan Normalisasi RMS
Catatan: setelah langkah ini bentuknya tidak berubah, nilainya hanya dinormalisasi.
<br>
Hal yang perlu diingat, kita memerlukan norm_eps (dari konfigurasi) karena kita tidak ingin secara tidak sengaja mengatur RMS ke 0 dan membagi dengan 0.
<br>
Berikut adalah rumusnya:
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/rms.png" width="600"/>
</div>

```python
# def rms_norm(tensor, norm_weights):
#     rms = (tensor.pow(2).mean(-1, keepdim=True) + norm_eps)**0.5
#     return tensor * (norm_weights / rms)
def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights
```

# Membangun Lapisan Pertama Transformer

### Normalisasi
Anda akan melihat saya mengakses layer.0 dari kamus model (ini adalah lapisan pertama).
<br>
Setelah dinormalisasi, bentuknya masih [17x4096] sama seperti embedding tetapi dinormalisasi.

<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/norm.png" width="600"/>
</div>

```python
token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"])
token_embeddings.shape
```

    torch.Size([17, 4096])

### Implementasi Perhatian dari Nol
Mari kita muat kepala perhatian dari lapisan pertama transformer.
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/qkv.png" width="600"/>
</div>

<br>

&gt; Saat kita memuat vektor query, key, value, dan output dari model, kita melihat bentuknya [4096x4096], [1024x4096], [1024x4096], [4096x4096].
<br>
&gt; Pada pandangan pertama ini aneh karena idealnya kita ingin setiap q, k, v, dan o untuk setiap kepala secara

 individu.
<br>
&gt; Penulis kode menggabungkannya bersama-sama karena ini membantu mempercepat perkalian kepala perhatian.
<br>
&gt; Saya akan membuka semuanya...

```python
print(
    model["layers.0.attention.wq.weight"].shape,
    model["layers.0.attention.wk.weight"].shape,
    model["layers.0.attention.wv.weight"].shape,
    model["layers.0.attention.wo.weight"].shape
)
```

    torch.Size([4096, 4096]) torch.Size([1024, 4096]) torch.Size([1024, 4096]) torch.Size([4096, 4096])

### Membuka Query
Pada bagian berikutnya, kita akan membuka query dari beberapa kepala perhatian, hasilnya adalah [32x128x4096].
<br><br>
Di sini, 32 adalah jumlah kepala perhatian di Llama3, 128 adalah ukuran vektor query, dan 4096 adalah ukuran embedding token.

```python
q_layer0 = model["layers.0.attention.wq.weight"]
head_dim = q_layer0.shape[0] // n_heads
q_layer0 = q_layer0.view(n_heads, head_dim, dim)
q_layer0.shape
```

    torch.Size([32, 128, 4096])

### Saya akan mengimplementasikan kepala pertama dari lapisan pertama
Di sini saya mengakses matriks bobot query kepala pertama dari lapisan pertama, ukuran matriks bobot query ini adalah [128x4096].

```python
q_layer0_head0 = q_layer0[0]
q_layer0_head0.shape
```

    torch.Size([128, 4096])

### Kita sekarang mengalikan bobot query dengan embedding token, untuk menerima query untuk token
Di sini Anda dapat melihat hasilnya adalah [17x128], ini karena kita memiliki 17 token dan untuk setiap token ada query dengan panjang 128.
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/q_per_token.png" width="600"/>
</div>

```python
q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T)
q_per_token.shape
```

    torch.Size([17, 128])

## Pengkodean Posisi
Kita sekarang berada pada tahap di mana kita memiliki vektor query untuk setiap token dalam prompt kita, tetapi jika Anda pikirkan lagi -- vektor query tersebut secara individu tidak memiliki informasi tentang posisinya dalam prompt.
<br><br>
query: "the answer to the ultimate question of life, the universe, and everything is "
<br><br>
Dalam prompt kita menggunakan "the" tiga kali, kita memerlukan vektor query dari ketiga token "the" untuk memiliki vektor query yang berbeda (masing-masing dengan ukuran [1x128]) berdasarkan posisi mereka dalam query. Kita melakukan rotasi ini menggunakan RoPE (rotory positional embedding).
<br><br>
### RoPE
Tonton video ini (ini yang saya tonton) untuk memahami matematika di baliknya.
[https://www.youtube.com/watch?v=o29P0Kpobz0&t=530s](https://www.youtube.com/watch?v=o29P0Kpobz0&t=530s)

<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/rope.png" width="600"/>
</div>

```python
q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
q_per_token_split_into_pairs.shape
```

    torch.Size([17, 64, 2])

Pada langkah di atas, kita membagi vektor query menjadi pasangan-pasangan, kita menerapkan rotasi sudut pada setiap pasangan!
<br><br>
Sekarang kita memiliki vektor dengan ukuran [17x64x2], ini adalah query dengan panjang 128 yang dibagi menjadi 64 pasangan untuk setiap token dalam prompt! Setiap dari 64 pasangan tersebut akan dirotasi dengan m*(theta) di mana m adalah posisi token untuk yang kita rotasi querynya!

<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/qsplit.png" width="600"/>
</div>

## Menggunakan Perkalian Titik dari Bilangan Kompleks untuk Merotasi Vektor
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/freq_cis.png" width="600"/>
</div>

```python
zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
zero_to_one_split_into_64_parts
```

    tensor([0.0000, 0.0156, 0.0312, 0.0469, 0.0625, 0.0781, 0.0938, 0.1094, 0.1250,
            0.1406, 0.1562, 0.1719, 0.1875, 0.2031, 0.2188, 0.2344, 0.2500, 0.2656,
            0.2812, 0.2969, 0.3125, 0.3281, 0.3438, 0.3594, 0.3750, 0.3906, 0.4062,
            0.4219, 0.4375, 0.4531, 0.4688, 0.4844, 0.5000, 0.5156, 0.5312, 0.5469,
            0.5625, 0.5781, 0.5938, 0.6094, 0.6250, 0.6406, 0.6562, 0.6719, 0.6875,
            0.7031, 0.7188, 0.7344, 0.7500, 0.7656, 0.7812, 0.7969, 0.8125, 0.8281,
            0.8438, 0.8594, 0.8750, 0.8906, 0.9062, 0.9219, 0.9375, 0.9531, 0.9688,
            0.9844])

```python
freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
freqs
```

    tensor([1.0000e+00, 8.1462e-01, 6.6360e-01, 5.4058e-01, 4.4037e-01, 3.5873e-01,
            2.9223e-01, 2.3805e-01, 1.9392e-01, 1.5797e-01, 1.2869e-01, 1.0483e-01,
            8.5397e-02, 6.9566e-02, 5.6670e-02, 4.6164e-02, 3.7606e-02, 3.0635e-02,
            2.4955e-02, 2.0329e-02, 1.6560e-02, 1.3490e-02, 1.0990e-02, 8.9523e-03,
            7.2927e-03, 5.9407e-03, 4.8394e-03, 3.9423e-03, 3.2114e-03, 2.6161e-03,
            2.1311e-03, 1.7360e-03, 1.4142e-03, 1.1520e-03, 9.3847e-04, 7.6450e-04,
            6.2277e-04, 5.0732e-04, 4.1327e-04, 3.3666e-04, 2.7425e-04, 2.2341e-04,
            1.8199e-04, 1.4825e-04, 1.2077e-04, 9.8381e-05, 8.0143e-05, 6.5286e-05,
            5.3183e-05, 4.3324e-05, 3.5292e-05, 2.8750e-05, 2.3420e-05, 1.9078e-05,
            1.5542e-05, 1.2660e-05, 1.0313e-05, 8.4015e-06, 6.8440e-06, 5.5752e-06,
            4.5417e-06, 3.6997e-06, 3.0139e-06, 2.4551e-06])

```python
freqs_for_each_token = torch.outer(torch.arange(17), freqs)
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
freqs_cis.shape



# melihat baris ketiga dari freqs_cis
value = freqs_cis[3]
plt.figure()
for i, element in enumerate(value[:17]):
    plt.plot([0, element.real], [0, element.imag], color='blue', linewidth=1, label=f"Index: {i}")
    plt.annotate(f"{i}", xy=(element.real, element.imag), color='red')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Plot of one row of freqs_cis')
plt.show()
```

![png](https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/implllama3_30_0.png)

### Sekarang kita memiliki bilangan kompleks (vektor perubahan sudut) untuk setiap elemen query token
Kita dapat mengubah query kita (yang telah kita bagi menjadi pasangan) sebagai bilangan kompleks dan kemudian melakukan perkalian titik untuk merotasi query berdasarkan posisi.
<br>
Sejujurnya, ini sangat indah untuk dipikirkan :)

```python
q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
q_per_token_as_complex_numbers.shape
```

    torch.Size([17, 64])

```python
q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis
q_per_token_as_complex_numbers_rotated.shape
```

    torch.Size([17, 64])

### Setelah vektor rotasi diperoleh
Kita dapat mengembalikan query sebagai pasangan dengan melihat bilangan kompleks sebagai bilangan riil lagi.

```python
q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers_rotated)
q_per_token_split_into_pairs_rotated.shape
```

    torch.Size([17, 64, 2])

Pasangan yang dirotasi sekarang digabungkan, kita sekarang memiliki vektor query baru (vektor query yang dirotasi) dengan ukuran [17x128] di mana 17 adalah jumlah token dan 128 adalah dimensi vektor query.

```python
q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
q_per_token_rotated.shape
```

    torch.Size([17, 128])

# Kunci (hampir sama dengan query)
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/keys.png" width="600px"/>
</div>
Saya sangat malas, jadi saya tidak akan menjelaskan matematika untuk kunci, yang perlu Anda ingat adalah:
<br>
&gt; Kunci menghasilkan vektor kunci juga dengan dimensi 128.
<br>
&gt; Kunci hanya memiliki 1/4 jumlah bobot dibandingkan query, ini karena bobot untuk kunci dibagi di antara 4 kepala sekaligus, untuk mengurangi jumlah perhitungan yang dibutuhkan.
<br>
&gt; Kunci juga dirotasi untuk menambahkan informasi posisi, seperti halnya query karena alasan yang sama.

```python
k_layer0 = model["layers.0.attention.wk.weight"]
k_layer0 = k_layer0.view(n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim)
k_layer0.shape
```

    torch.Size([8, 128, 4096])

```python
k_layer0_head0 = k_layer0[0]
k_layer0_head0.shape
```

    torch.Size([128, 4096])

```python
k_per_token = torch.matmul(token_embeddings, k_layer0_head0.T)
k_per_token.shape
```

    torch.Size([17, 128])

```python
k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
k_per_token_split_into_pairs.shape
```

    torch.Size([17, 64, 2])

```python
k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
k_per_token_as_complex_numbers.shape
```

    torch.Size([17, 64])

```python
k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
k_per_token_split_into_pairs_rotated.shape
```

    torch.Size([17, 64, 2])

```python
k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
k_per_token_rotated.shape
```

    torch.Size([17, 128])

## Pada tahap ini sekarang kita memiliki nilai query dan kunci yang dirotasi untuk setiap token.
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/keys0.png" width="600px"/>
</div>
Masing-masing query dan kunci sekarang berukuran [17x128].

## Pada langkah berikutnya, kita akan mengalikan matriks query dan kunci
Melakukan ini akan memberikan kita skor yang memetakan setiap token dengan satu sama lain.
<br>
Skor ini menggambarkan seberapa baik setiap query token berhubungan dengan setiap kunci token.
INI ADALAH SELF ATTENTION :)
<br>
Ukuran matriks skor perhatian (qk_per_token) adalah [17x17] di mana 17 adalah jumlah token dalam prompt.

<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/qkmatmul.png" width="600px"/>
</div>

```python
qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(head_dim)**0.5
qk_per_token.shape
```

    torch.Size([17, 17])

# Kita sekarang harus memasker skor query kunci
Selama proses pelatihan Llama3, skor qk token masa depan dimasker.
<br>
Kenapa? Karena selama pelatihan kita hanya belajar memprediksi token menggunakan token sebelumnya.
<br>
Sebagai hasilnya, selama inferensi kita mengatur token masa depan ke nol.
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/mask.png" width="600px"/>
</div>

```python
def display_qk_heatmap(qk_per_token):
    _, ax = plt.subplots()
    im = ax.imshow(qk_per_token.to(float).detach(), cmap='viridis')
    ax.set_xticks(range(len(prompt_split_as_tokens)))
    ax.set_yticks(range(len(prompt_split_as_tokens)))
    ax.set_xticklabels(prompt_split_as_tokens)
    ax.set_yticklabels(prompt_split_as_tokens)
    ax.figure.colorbar(im, ax=ax)

display_qk_heatmap(qk_per_token)
```

![png](https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/implllama3_50_0.png)

```python
mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
mask = torch.triu(mask, diagonal=1)
mask
```

    tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0.,

 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])


```python
qk_per_token_after_masking = qk_per_token + mask
display_qk_heatmap(qk_per_token_after_masking)
```

![png](https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/implllama3_52_0.png)

<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/softmax.png" width="600px"/>
</div>

```python
qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
display_qk_heatmap(qk_per_token_after_masking_after_softmax)
```

![png](https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/implllama3_54_0.png)

## Nilai (hampir akhir dari perhatian)
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/value.png" width="600px"/>
</div>
Skor ini (0-1) digunakan untuk menentukan seberapa banyak dari matriks nilai yang digunakan per token.
<br>
&gt; Seperti halnya kunci, bobot nilai juga dibagi di antara setiap 4 kepala perhatian (untuk menghemat perhitungan).
<br>
&gt; Akibatnya, ukuran matriks bobot nilai di bawah ini adalah [8x128x4096].

```python
v_layer0 = model["layers.0.attention.wv.weight"]
v_layer0 = v_layer0.view(n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim)
v_layer0.shape
```

    torch.Size([8, 128, 4096])

Matriks bobot nilai kepala pertama lapisan pertama diberikan di bawah ini.

```python
v_layer0_head0 = v_layer0[0]
v_layer0_head0.shape
```

    torch.Size([128, 4096])

## Vektor Nilai
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/v0.png" width="600px"/>
</div>
Kita sekarang menggunakan bobot nilai untuk mendapatkan nilai perhatian per token, ini berukuran [17x128] di mana 17 adalah jumlah token dalam prompt dan 128 adalah dimensi vektor nilai per token.

```python
v_per_token = torch.matmul(token_embeddings, v_layer0_head0.T)
v_per_token.shape
```

    torch.Size([17, 128])

## Perhatian
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/attention.png" width="600px"/>
</div>
Vektor perhatian hasil setelah dikalikan dengan nilai per token adalah berukuran [17x128].

```python
qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
qkv_attention.shape
```

    torch.Size([17, 128])

# Perhatian Multi-Kepala
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/heads.png" width="600px"/>
</div>
KITA SEKARANG MEMILIKI NILAI PERHATIAN DARI LAPISAN PERTAMA DAN KEPALA PERTAMA.
<br>
Sekarang saya akan menjalankan loop dan melakukan perhitungan yang sama seperti pada sel-sel sebelumnya tetapi untuk setiap kepala di lapisan pertama.

```python
qkv_attention_store = []

for head in range(n_heads):
    q_layer0_head = q_layer0[head]
    k_layer0_head = k_layer0[head//4] # bobot kunci dibagi di antara 4 kepala
    v_layer0_head = v_layer0[head//4] # bobot nilai dibagi di antara 4 kepala
    q_per_token = torch.matmul(token_embeddings, q_layer0_head.T)
    k_per_token = torch.matmul(token_embeddings, k_layer0_head.T)
    v_per_token = torch.matmul(token_embeddings, v_layer0_head.T)

    q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
    q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
    q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

    k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
    k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
    k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
    mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
    mask = torch.triu(mask, diagonal=1)
    qk_per_token_after_masking = qk_per_token + mask
    qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention_store.append(qkv_attention)

len(qkv_attention_store)
```

    32

<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/stacked.png" width="600px"/>
</div>
Kita sekarang memiliki matriks qkv_attention untuk semua 32 kepala di lapisan pertama. Selanjutnya, saya akan menggabungkan semua skor perhatian menjadi satu matriks besar dengan ukuran [17x4096].
<br>
Kita hampir selesai :)

```python
stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
stacked_qkv_attention.shape
```

    torch.Size([17, 4096])

# Matriks Bobot, Salah Satu Langkah Terakhir
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/weightmatrix.png" width="600px"/>
</div>
Salah satu hal terakhir yang perlu dilakukan untuk lapisan perhatian 0 adalah mengalikan matriks bobot dari lapisan.

```python
w_layer0 = model["layers.0.attention.wo.weight"]
w_layer0.shape
```

    torch.Size([4096, 4096])

### Ini adalah lapisan linear sederhana, jadi kita cukup melakukan perkalian matriks

```python
embedding_delta = torch.matmul(stacked_qkv_attention, w_layer0.T)
embedding_delta.shape
```

    torch.Size([17, 4096])

<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/afterattention.png" width="600px"/>
</div>
Kita sekarang memiliki perubahan dalam nilai embedding setelah perhatian, yang harus ditambahkan ke embedding token asli.

```python
embedding_after_edit = token_embeddings_unnormalized + embedding_delta
embedding_after_edit.shape
```

    torch.Size([17, 4096])

## Kita normalisasi dan kemudian menjalankan jaringan saraf feedforward melalui embedding delta
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/norm_after.png" width="600px"/>
</div>

```python
embedding_after_edit_normalized = rms_norm(embedding_after_edit, model["layers.0.ffn_norm.weight"])
embedding_after_edit_normalized.shape
```

    torch.Size([17, 4096])

## Memuat Bobot FF dan Mengimplementasikan Jaringan Saraf Feedforward
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/swiglu.png" width="600px"/>
</div>
Di Llama3, mereka menggunakan jaringan feedforward SwiGLU, arsitektur jaringan ini sangat baik dalam menambahkan non-linearitas saat dibutuhkan oleh model.
<br>
Saat ini sudah menjadi standar untuk menggunakan arsitektur jaringan feedforward ini di

 LLMs.

```python
w1 = model["layers.0.feed_forward.w1.weight"]
w2 = model["layers.0.feed_forward.w2.weight"]
w3 = model["layers.0.feed_forward.w3.weight"]
output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
output_after_feedforward.shape
```

    torch.Size([17, 4096])

# KITA AKHIRNYA MEMILIKI EMBEDDING BARU YANG DIEDIT UNTUK SETIAP TOKEN SETELAH LAPISAN PERTAMA
Hanya 31 lapisan lagi yang perlu diimplementasikan (satu loop lagi)
<br>
Anda bisa membayangkan embedding yang diedit ini memiliki informasi tentang semua query yang diajukan pada lapisan pertama.
<br>
Sekarang setiap lapisan akan mengodekan query yang semakin kompleks, sampai kita memiliki embedding yang mengetahui segala sesuatu tentang token berikutnya yang kita butuhkan.

```python
layer_0_embedding = embedding_after_edit + output_after_feedforward
layer_0_embedding.shape
```

    torch.Size([17, 4096])

# Segalanya Sekaligus
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/god.png" width="600px"/>
</div>
Ya, ini dia. Segala sesuatu yang telah kita lakukan sebelumnya, semuanya sekaligus, untuk setiap lapisan.
<br>
# Selamat membaca :)

```python
final_embedding = token_embeddings_unnormalized
for layer in range(n_layers):
    qkv_attention_store = []
    layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
    q_layer = model[f"layers.{layer}.attention.wq.weight"]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
    k_layer = model[f"layers.{layer}.attention.wk.weight"]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
    v_layer = model[f"layers.{layer}.attention.wv.weight"]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    for head in range(n_heads):
        q_layer_head = q_layer[head]
        k_layer_head = k_layer[head//4]
        v_layer_head = v_layer[head//4]
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
        mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        qkv_attention_store.append(qkv_attention)

    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
    embedding_after_edit = final_embedding + embedding_delta
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
    w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
    w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
    w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
    output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
    final_embedding = embedding_after_edit + output_after_feedforward
```

# Kita sekarang memiliki embedding akhir, tebakan terbaik model tentang token berikutnya
Ukuran embedding tetap sama seperti embedding token reguler [17x4096] di mana 17 adalah jumlah token dan 4096 adalah dimensi embedding.
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/last_norm.png" width="600px"/>
</div>

```python
final_embedding = rms_norm(final_embedding, model["norm.weight"])
final_embedding.shape
```

    torch.Size([17, 4096])

# Akhirnya, mari kita dekode embedding menjadi nilai token
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/finallayer.png" width="600px"/>
</div>
Kita akan menggunakan output decoder untuk mengonversi embedding akhir menjadi token.

```python
model["output.weight"].shape
```

    torch.Size([128256, 4096])

# Kita menggunakan embedding token terakhir untuk memprediksi nilai berikutnya
Mudah-mudahan dalam kasus kita, 42 :)
Catatan: 42 adalah jawaban dari "the answer to the ultimate question of life, the universe, and everything is", menurut buku "Hitchhiker's Guide to the Galaxy", sebagian besar LLM modern akan menjawab dengan 42 di sini, yang seharusnya memvalidasi seluruh kode kita! Semoga berhasil :)

```python
logits = torch.matmul(final_embedding[-1], model["output.weight"].T)
logits.shape
```

    torch.Size([128256])

### Model memprediksi token nomor 2983 sebagai token berikutnya, apakah ini nomor token untuk 42?
Saya memotivasi Anda, ini adalah sel kode terakhir, semoga Anda menikmati :)

```python
next_token = torch.argmax(logits, dim=-1)
next_token
```

    tensor(2983)

# Mari kita pergi
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/42.png" width="600px"/>
</div>

```python
tokenizer.decode([next_token.item()])
```

    '42'

# Terima kasih, saya mencintaimu :)
Ini adalah akhirnya. Semoga Anda menikmati membaca ini!

Jika Anda ingin mendukung karya saya:

1. Ikuti saya di Twitter [https://twitter.com/naklecha](https://twitter.com/naklecha).
2. Atau, belikan saya kopi [https://www.buymeacoffee.com/naklecha](https://www.buymeacoffee.com/naklecha).

Jujur saja, jika Anda berhasil sejauh ini, Anda sudah membuat hari saya lebih baik :)

## Apa yang memotivasi saya?
Teman-teman saya dan saya memiliki misi - untuk membuat penelitian lebih mudah diakses!
Kami menciptakan sebuah lab penelitian bernama A10 - [AAAAAAAAAA.org](http://aaaaaaaaaa.org/)

Twitter A10 - [https://twitter.com/aaaaaaaaaaorg](https://twitter.com/aaaaaaaaaaorg)

tesis kami:
<div>
    <img src="https://raw.githubusercontent.com/naklecha/llama3-from-scratch/main/images/a10.png" width="600px"/>
</div>