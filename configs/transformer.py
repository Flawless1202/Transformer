name = "Transformer"
version = "0.1.0"
model = dict(
    d_word_vec=512,
    d_model=512,
    d_hid=2048,
    n_layers=6,
    n_heads=8,
    d_k=64,
    d_v=64,
    dropout=0.1,
    trg_emb_prj_weight_sharing=True,
    emb_src_trg_weight_sharing=True)
data = dict(
    data_path="data/m30k_deen_shr.pkl",
    batch_size=128)
train_cfg = dict(
    smoothing=True)
lr_cfg = dict(
    init_lr=2.,
    warmup_steps=128000)
random_seed = 123456
num_gpus = 1
max_epochs = 400
checkpoint_path = "work_dirs/checkpoints"
log_path = "work_dirs/logs"
load_from_checkpoint = None
resume_from_checkpoint = None
batch_size_times = 2
simple_profiler = True
