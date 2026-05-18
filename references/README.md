# References

This directory stores the primary sources that justify the project's design choices.

## Layout

```text
references/
  papers/         PDFs or links to primary papers and standards
  notes/          our own summaries and implementation takeaways
  citations.bib   shared bibliography
```

## Keep a reference only if it helps answer one of these questions

- What architecture are we using?
- What dataset or benchmark are we following?
- What evaluation standard matters?
- What prior work already solves part of our problem?
- What claim in our docs or experiments needs support?

## Working rule

Every important paper should eventually have:

1. the paper itself in `papers/` or a source note pointing to it,
2. a short summary in `notes/`,
3. an entry in `citations.bib`.
