from rt.main import main
from rt.tasks import all_tasks, forecast_tasks

if __name__ == "__main__":
    main(
        # misc
        project="rt",
        eval_splits=["val", "test"],
        eval_freq=1_000,
        eval_pow2=True,
        max_eval_steps=40,
        load_ckpt_path=None,
        save_ckpt_dir="ckpts/leave_rel-amazon",
        compile_=True,
        seed=0,
        # data
        train_tasks=[t for t in all_tasks if t[0] != "rel-amazon"],
        eval_tasks=[t for t in forecast_tasks if t[0] == "rel-amazon"],
        batch_size=32,
        num_workers=8,
        max_bfs_width=256,
        # optimization
        lr=1e-3,
        wd=0.1,
        lr_schedule=True,
        max_grad_norm=1.0,
        max_steps=50_001,
        # model
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seq_len=1024,
        num_blocks=12,
        d_model=256,
        num_heads=8,
        d_ff=1024,
    )
