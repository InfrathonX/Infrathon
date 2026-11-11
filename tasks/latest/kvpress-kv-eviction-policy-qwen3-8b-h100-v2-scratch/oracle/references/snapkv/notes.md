Version 1 is a SnapKV-style policy (observation-window attention): the last
64 context queries, rotated with the supplied RoPE tables, score every cached
key; attention mass is averaged over the window and the query heads of a KV
head, smoothed by 1-D pooling (width 7); four sinks and the window are kept.
