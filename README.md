# RCWA4D

This is an electromagnetic solver based on Rigorous Coupled Wave Analyses (RCWA) for layered structures with incommensurate periodicities. It can be used to obtain the scattering properties (e.g. transmission, reflection, diffractions) in such structures.

The theory and preliminary results of this method are described in [this paper first](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.136101) and [this paper later](https://www.sciencedirect.com/science/article/abs/pii/S0010465524002790).

The results obtained from this method has matched well to experiments in [microwave domain](https://www.science.org/doi/10.1126/sciadv.add4339) and [optical domain](https://www.science.org/doi/10.1126/sciadv.adh8498).

This package has also been used for applications such as [tunable filter](https://pubs.acs.org/doi/10.1021/acsphotonics.1c01263).



### If you find this package helpful for your research, please consider citing:

*(the companion paper for this repo)*

`@article{LOU2025109356,
  title = {RCWA4D: Electromagnetic solver for layered structures with incommensurate periodicities},
  journal = {Computer Physics Communications},
  volume = {306},
  pages = {109356},
  year = {2025},
  issn = {0010-4655},
  doi = {https://doi.org/10.1016/j.cpc.2024.109356},
  url = {https://www.sciencedirect.com/science/article/pii/S0010465524002790},
  author = {Beicheng Lou and Shanhui Fan},
  keywords = {Computational electromagnetics, Rigorous coupled wave analysis, Twisted bilayer photonic crystal slabs, Moire optics, Quasi-periodic structure},
  abstract = {We describe RCWA4D, an electromagnetic solver for layered structures with incommensurate periodicities. Our method is based on an extension of the rigorous coupled wave analysis. We illustrate our method on the example of twisted bilayer photonic crystal and show that various properties of such structures can be reliably simulated. The method can be generalized to multi-layer structures in general in which each layer is periodic or quasi-periodic.}
}`

*(the first paper on the algorithm)*

`@article{PhysRevLett.126.136101,
  title = {Theory for Twisted Bilayer Photonic Crystal Slabs},
  author = {Lou, Beicheng and Zhao, Nathan and Minkov, Momchil and Guo, Cheng and Orenstein, Meir and Fan, Shanhui},
  journal = {Phys. Rev. Lett.},
  volume = {126},
  issue = {13},
  pages = {136101},
  numpages = {6},
  year = {2021},
  month = {Mar},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.126.136101},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.126.136101}
}`

*(the first experimental verification, done in microwave)*

`@article{sciadv.add4339,
  author = {Beicheng Lou  and Benjamin Wang  and Jesse A Rodríguez  and Mark Cappelli  and Shanhui Fan },
  title = {Tunable guided resonance in twisted bilayer photonic crystal},
  journal = {Science Advances},
  volume = {8},
  number = {48},
  pages = {eadd4339},
  year = {2022},
  doi = {10.1126/sciadv.add4339},
  URL = {https://www.science.org/doi/abs/10.1126/sciadv.add4339},
  eprint = {https://www.science.org/doi/pdf/10.1126/sciadv.add4339}
}`

*(the first experimental verification in optical frequency, with very comprehensive comparisons)*

`@article{sciadv.adh8498,
  author = {Haoning Tang  and Beicheng Lou  and Fan Du  and Mingjie Zhang  and Xueqi Ni  and Weijie Xu  and Rebekah Jin  and Shanhui Fan  and Eric Mazur },
  title = {Experimental probe of twist angle–dependent band structure of on-chip optical bilayer photonic crystal},
  journal = {Science Advances},
  volume = {9},
  number = {28},
  pages = {eadh8498},
  year = {2023},
  doi = {10.1126/sciadv.adh8498},
  URL = {https://www.science.org/doi/abs/10.1126/sciadv.adh8498},
  eprint = {https://www.science.org/doi/pdf/10.1126/sciadv.adh8498}
}`

as well as a series of applications:

*(tunable frequency filter)*

`@article{lou_tunable_2022,
  title = {Tunable Frequency Filter Based on Twisted Bilayer Photonic Crystal Slabs},
  volume = {9},
  url = {https://doi.org/10.1021/acsphotonics.1c01263},
  doi = {10.1021/acsphotonics.1c01263},
  pages = {800--805},
  number = {3},
  journaltitle = {{ACS} Photonics},
  shortjournal = {{ACS} Photonics},
  author = {Lou, Beicheng and Fan, Shanhui},
  date = {2022-03-16},
  note = {Publisher: American Chemical Society},
}`

*(tunable thermal emission)*

`@article{guo_wide_2021,
    author = {Guo, Cheng and Guo, Yu and Lou, Beicheng and Fan, Shanhui},
    title = "{Wide wavelength-tunable narrow-band thermal radiation from moiré patterns}",
    journal = {Applied Physics Letters},
    volume = {118},
    number = {13},
    pages = {131111},
    year = {2021},
    month = {03},
    issn = {0003-6951},
    doi = {10.1063/5.0047308},
    url = {https://doi.org/10.1063/5.0047308}
}`

*(reconfigurable optical singularity)*

`@article{ni2024three,
  title={Three-Dimensional Reconfigurable Optical Singularities in Bilayer Photonic Crystals},
  author={Ni, Xueqi and Liu, Yuan and Lou, Beicheng and Zhang, Mingjie and Hu, Evelyn L and Fan, Shanhui and Mazur, Eric and Tang, Haoning},
  journal={Physical Review Letters},
  volume={132},
  number={7},
  pages={073804},
  year={2024},
  publisher={APS}
}`

*(smart adaptive sensing)*

`@article{tang2023chip,
  title={On-Chip Multidimensional Dynamic Control of Twisted Moir$\backslash$'e Photonic Crystal for Smart Sensing and Imaging},
  author={Tang, Haoning and Lou, Beicheng and Du, Fan and Gao, Guangqi and Zhang, Mingjie and Ni, Xueqi and Hu, Evelyn and Yacoby, Amir and Cao, Yuan and Fan, Shanhui and others},
  journal={arXiv preprint arXiv:2312.09089},
  year={2023}
}`



For collaborations and inquiries, please contact beichenglou@stanford.edu.

Also, feel free to reach out for feature requests. This repo is casually but actively maintained : )
