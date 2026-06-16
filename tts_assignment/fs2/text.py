"""Phoneme vocabulary for IPA-compressed LJSpeech manifest."""

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PHONEMES = [
    "SIL",
    "aɪ", "aʊ", "b", "d", "dʒ", "eɪ", "f", "h", "i", "j",
    "k", "l", "m", "n", "oʊ", "p", "r", "s", "t", "tʃ",
    "u", "v", "w", "z", "æ", "ð", "ŋ", "ɑ", "ɔ", "ɔr",
    "ɔɪ", "ə", "ər", "ɛ", "ɛr", "ɡ", "ɪ", "ɪr", "ʃ", "ʊ",
    "ʊr", "ʌ", "ʌr", "ʒ", "θ",
]

SYMBOLS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + PHONEMES

PHONE_TO_ID = {p: i for i, p in enumerate(SYMBOLS)}
ID_TO_PHONE = {i: p for i, p in enumerate(SYMBOLS)}

PAD_ID = PHONE_TO_ID[PAD_TOKEN]
BOS_ID = PHONE_TO_ID[BOS_TOKEN]
EOS_ID = PHONE_TO_ID[EOS_TOKEN]
UNK_ID = PHONE_TO_ID[UNK_TOKEN]

VOCAB_SIZE = len(SYMBOLS)


def phonemes_to_ids(phonemes):
    return [PHONE_TO_ID.get(p, UNK_ID) for p in phonemes]


def ids_to_phonemes(ids):
    return [ID_TO_PHONE.get(i, UNK_TOKEN) for i in ids]
