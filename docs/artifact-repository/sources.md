# Sources

`artifacts.sources`: a named body of raw text at `sources/<name>/body.txt`.
Everything downstream reads only that file, so a subclass only has to say where
the text comes from.

| Class | Field | Meaning |
| --- | --- | --- |
| `Source` | `name` | The folder name and the uid. Abstract. |
| `SourceURL` | `url` | Where to download the text from. |

```python
SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
```

## The job

`SourceURLJob` streams the response into `body.txt` through
`worker.publishing`, a megabyte at a time, so a corpus of gigabytes is never
held in memory. A URL ending in `.gz` is unzipped in the same pass, and must be
one complete gzip stream. A response served as `text/html` is refused: that is
a login wall or a 404 page served with a 200. Progress reports bytes off the
wire.

A source has no dependencies, so it is the root of every chain.
