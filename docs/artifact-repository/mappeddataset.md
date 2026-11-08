# Mapped dataset

`artifacts.mappeddataset.MappedDataSet`: a training set that owns no bytes. It
has no job (`producer` is `None`); it is done exactly when its tokenized
sources are, and its bound object reads like one array over their own
`tokens.bin` files, with `<|endoftext|>` between sources. Its folder
`mappeddatasets/<uid>/` holds only the manifest.

Fields, ordering, validation, `from_sources` and the bound `train` and `valid`
splits are those of [Dataset](dataset.md). It additionally refuses a
set with no sources in either split, since with nothing to be done by it would
read as done before anything existed.

## TokenStream

`artifacts.mappeddataset.tokenstream.TokenStream` is what `train.tokens` and
`valid.tokens` are once bound: one memmap per file, a separator between
consecutive files, and each piece's offset worked out once. A window inside one
file is that memmap's own slice; a window spanning a boundary is copied, that
window only. Only step-1 slices and integer indexes are served.

## Why

Concatenating gigabytes into a second copy costs time and disk for no new
information. The trade is that a mapped dataset's completeness is borrowed, so
declaration can show it as `new` while every file it reads already exists.
