"""
standard template and objects for structure prediction
"""

import pathlib as pl
import numpy as np
import numpy.typing as npt
from ...chain import Chain
from ...constants import atom_order
from .utils import reindex_chains, reindex_residues, validate_array_range
from pydantic import field_validator
from .base import FoldingOracle, FoldingResult
from typing import List, Any, Type
from boileroom.models.esm.esmfold import ESMFoldOutput  # type: ignore
from boileroom.models.esm.esmfold import ESMFold as ESMFoldBoiler

from biotite.structure import AtomArray


class ESMFoldResult(FoldingResult):
    """
    Stores statistics from the ESMFold folding algorithm.
    """

    input_chains: list[Chain]
    structure: AtomArray  # structure of the predicted model
    local_plddt: npt.NDArray[np.float64]  # local ( per residue ) predicted LDDT score (0 to 1)
    ptm: npt.NDArray[np.float64]  # (global) predicted template modelling score (0 to 1)
    pae: npt.NDArray[np.float64]  # pairwise predicted alignment error

    # for ptm: see Zhang Y and Skolnick J (2004). "Scoring function for automated assessment of
    # protein structure template quality". Proteins. 57 (4): 702–710. doi:10.1002/prot.20264

    @field_validator('local_plddt')
    def validate_local_plddt(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'local_plddt', 0, 1)

    @field_validator('ptm')
    def validate_ptm(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'ptm', 0, 1)

    def save_attributes(self, filepath: pl.Path) -> None:
        np.savetxt(filepath.with_suffix('.plddt'), self.local_plddt[0], fmt='%.6f', header='plddt')
        np.savetxt(filepath.with_suffix('.pae'), self.pae[0], fmt='%.6f', header='pae')


class ESMFold(FoldingOracle):
    """
    Object that uses ESMFold to predict structure of proteins from sequence.

    WIP: For now we will be using ModalFold to do this reliably without much env issues.
    """

    result_class: Type[ESMFoldResult] = ESMFoldResult

    def __init__(self, backend: str = 'modal', device: str | None = None, config: dict[str, Any] = {}):
        """
        Initialize ESMFold oracle.

        Parameters
        ----------
        backend : str
            Backend to use. Supported values: "modal", "apptainer", "apptainer:<image-tag>".
        device : str | None
            Device to use (e.g., "cuda:0", "cuda:1").
        config : dict[str, Any]
            Configuration dictionary passed to the model
        """
        self._validate_backend(backend)
        self.backend = backend
        self.device = device
        self.default_config = {
            'output_pdb': False,
            'output_cif': False,
            'output_atomarray': True,
            'glycine_linker': '',
            'position_ids_skip': 512,
        }
        # Always request these fields in the output
        self.required_fields = ['plddt', 'pae', 'ptm']
        self._load(config)

    def _load(self, config: dict[str, Any] = {}) -> None:
        config = {**self.default_config, **config}
        self.model = ESMFoldBoiler(backend=self.backend, device=self.device, config=config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Pre-process the sequence to be passed to the model for folding.
        Here, we assume, that we are using HuggingFace's implementation of ESMFold.
        Therefore, individual chains are separated by a ":" character.
        """
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def fold(self, chains: List[Chain]) -> ESMFoldResult:
        """
        Fold a list of chains using ESMFold.
        """
        sequences = self._pre_process(chains)
        options = {'include_fields': self.required_fields}
        output = self.model.fold(sequences, options=options)
        return self._reduce_output(output, chains)

    def _reduce_output(self, output: ESMFoldOutput, chains: List[Chain]) -> ESMFoldResult:
        """
        Reduce ESMFoldOutput (from boileroom.esmfold) to a ESMFoldResult object.
        In principle, any other metric from ESMFoldOutput can be passed down into the ESMFoldResult object.
        For instance, one could pass the distogram_logits to create an EnergyTerm related to that.
        """
        if output.atom_array is None or len(output.atom_array) == 0:
            raise ValueError('ESMFold output does not contain atom_array')
        atoms = output.atom_array
        atoms = reindex_chains(atoms, [chain.chain_ID for chain in chains])
        atoms = reindex_residues(atoms, chains)

        # These fields should always be present since we requested them via include_fields.
        # boileroom's Apptainer backend round-trips them through JSON as one already-processed
        # sample per input (see ESMFoldOutput.__post_init__ / _normalize_esmfold_plddt): plddt
        # arrives CA-only and unit-scale already, not the raw (batch, residue, atom37) tensor
        # this code used to assume. `_first_sample` also tolerates the old raw-tensor shape so
        # this keeps working if a future/older boileroom build ever sends that instead. There is
        # always exactly one sample per call (`_pre_process` returns a single joined sequence).
        local_plddt = self._first_sample(output.plddt, 'plddt')
        if local_plddt.ndim == 2 and local_plddt.shape[-1] == 37:
            local_plddt = local_plddt[:, atom_order['CA']]
        if local_plddt.ndim == 1:
            local_plddt = local_plddt[None, :]

        ptm = np.atleast_1d(self._first_sample(output.ptm, 'ptm'))
        if ptm.ndim == 1:
            ptm = ptm[None, :]

        pae = self._first_sample(output.pae, 'pae')
        if pae.ndim == 2:
            pae = pae[None, :, :]

        results = self.result_class(
            input_chains=chains,
            structure=atoms,
            local_plddt=local_plddt,
            ptm=ptm,
            pae=pae,
        )
        return results

    @staticmethod
    def _first_sample(value: Any, name: str) -> npt.NDArray[np.float64]:
        """Pull out the sole sample's array from an ESMFoldOutput field. `value[0]` works
        uniformly whether `value` is a per-sample list (the current boileroom contract) or a
        raw batch-first ndarray (an older/hypothetical one) -- both index their first axis the
        same way, and there is always exactly one sample per call (`_pre_process` returns a
        single joined sequence)."""
        if value is None or len(value) == 0:
            raise ValueError(f'ESMFold output does not contain {name} (requested via include_fields)')
        sample = value[0]
        if sample is None:
            raise ValueError(f'ESMFold output does not contain {name} (requested via include_fields)')
        return np.asarray(sample, dtype=np.float64)
