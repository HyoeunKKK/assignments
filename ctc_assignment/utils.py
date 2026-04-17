BLANK_IDX = 0
VOCAB = ['<blank>'] + list('abcdefghijklmnopqrstuvwxyz') + ["'", ' ']
# blank=0, a=1..z=26, '=27, space=28
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)  # 29


def encode(text: str) -> list[int]:
    return [CHAR2IDX[c] for c in text.lower() if c in CHAR2IDX]


def decode(indices) -> str:
    result = []
    prev = None
    for idx in indices:
        idx = int(idx)
        if idx != prev:
            if idx != BLANK_IDX:
                result.append(VOCAB[idx])
            prev = idx
    return ''.join(result)


def _edit_distance(hyp: list, ref: list) -> int:
    n, m = len(hyp), len(ref)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m]


def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    total_words = 0
    total_errors = 0
    for hyp, ref in zip(hypotheses, references):
        ref_words = ref.split()
        total_words += len(ref_words)
        total_errors += _edit_distance(hyp.split(), ref_words)
    return total_errors / max(total_words, 1)
