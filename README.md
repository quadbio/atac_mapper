# Atac Mapper

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/quadbio/atac_mapper/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/atac_mapper

Map query scATAC-seq data to a single-cell chromatin accessibility atlas using cisTopics.

## Getting started

Please refer to the [documentation][https://atac-mapper.readthedocs.io/en/latest/index.html],
in particular, the [API documentation][https://atac-mapper.readthedocs.io/en/latest/api.html].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install atac_mapper:

<!--
1) Install the latest release of `atac_mapper` from [PyPI][]:

```bash
pip install atac_mapper
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/quadbio/atac_mapper.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

If you find AtacMapper useful in your research, please, cite the following article:

```bibtex
@article{azbukina2025multiomic,
  title={Multi-omic human neural organoid cell atlas of the posterior brain},
  author={Azbukina, Nadezhda and He, Zhisong and Lin, Hsiu-Chuan and Santel, Malgorzata and Kashanian, Bijan and Maynard, Ashley and T{\"o}r{\"o}k, Tivadar and Okamoto, Ryoko and Nikolova, Marina and Kanton, Sabina and Br{\"o}samle, Valentin and Holtackers, Rene and Camp, J Gray and Treutlein, Barbara},
  journal={bioRxiv},
  pages={2025--03},
  year={2025},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.03.20.644368}
}
```

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/quadbio/atac_mapper/issues
[tests]: https://github.com/quadbio/atac_mapper/actions/workflows/test.yaml
[documentation]: https://atac_mapper.readthedocs.io
[changelog]: https://atac_mapper.readthedocs.io/en/latest/changelog.html
[api documentation]: https://atac_mapper.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/atac_mapper

<!-- Example of additional links you might want to add -->
[bioRxiv paper]: https://www.biorxiv.org/content/10.1101/2025.03.20.644368v1
[quadbio lab]: https://github.com/quadbio
[scanpy]: https://scanpy.readthedocs.io/
[scvi-tools]: https://docs.scvi-tools.org/
