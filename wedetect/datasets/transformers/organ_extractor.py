"""Pipeline transform that adds clinical-known organ_id to results dict.

Parses the image path (the path-derived signal `organ_*` segment, not GT
annotations) so this is permissible at inference time. Adds:

  results['organ_id']:   int in [0, num_organs)
  results['organ_name']: str

PackDetInputs.meta_keys must include these two keys for them to reach
data_sample.metainfo during inference. See config:

  dict(type='PackDetInputs',
       meta_keys=(..., 'organ_id', 'organ_name'))

Failure mode: if no organ segment is found in img_path, raises ValueError
(deliberate: silent-default-to-zero would mask data bugs).
"""
import json
from pathlib import Path

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class OrganExtractor:
    """Extract clinical organ id from the image path.

    Args:
        taxonomy_path: path to tct_ngc_taxonomy.json (provides organ_to_id).
        organ_segment_map: optional explicit mapping {path_token: organ_name}.
            By default each organ_name is matched as its space-to-underscore
            form (e.g. 'respiratory tract' → 'respiratory_tract').
        strict: if True, raise on unmatched paths. If False, write organ_id=-1.
    """

    def __init__(self,
                 taxonomy_path: str,
                 organ_segment_map: dict = None,
                 strict: bool = True):
        tax = json.loads(Path(taxonomy_path).read_text())
        self.organ_to_id = dict(tax['organ_to_id'])
        self.strict = strict

        if organ_segment_map is None:
            organ_segment_map = {
                organ.replace(' ', '_'): organ
                for organ in self.organ_to_id
            }
        # Uniqueness assertion: no path token should be a substring of another
        # (otherwise the loop's "first segment match" rule becomes order-
        # sensitive and fragile if a new organ is added). The 5 current
        # TCT_NGC organs satisfy this; assert so future taxonomy edits fail
        # fast rather than silently rerouting samples.
        tokens = sorted(organ_segment_map.keys(), key=len)
        for i, t in enumerate(tokens):
            for u in tokens[i + 1:]:
                if t in u or u in t:
                    raise ValueError(
                        f'OrganExtractor: organ tokens {t!r} and {u!r} are '
                        f'prefix/substring of each other — path matching is '
                        f'ambiguous. Provide explicit organ_segment_map.')
        self.segment_to_organ = organ_segment_map  # path token → canonical name

    def __call__(self, results: dict) -> dict:
        img_path = results.get('img_path') or results.get('filename')
        if img_path is None:
            if self.strict:
                raise ValueError('OrganExtractor: no img_path in results')
            results['organ_id'] = -1
            results['organ_name'] = ''
            return results

        organ_name = None
        for seg in str(img_path).split('/'):
            if seg in self.segment_to_organ:
                organ_name = self.segment_to_organ[seg]
                break

        if organ_name is None:
            if self.strict:
                raise ValueError(
                    f'OrganExtractor: no organ segment in img_path: {img_path}\n'
                    f'  expected one of: {sorted(self.segment_to_organ.keys())}')
            results['organ_id'] = -1
            results['organ_name'] = ''
            return results

        results['organ_id'] = self.organ_to_id[organ_name]
        results['organ_name'] = organ_name
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"organs={list(self.organ_to_id.keys())}, strict={self.strict})")
