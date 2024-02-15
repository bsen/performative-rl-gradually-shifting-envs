from typing import Type

from sperl.metapolicies.abstract_meta_policy import MetaPolicy
from sperl.metapolicies.algorithms.delayed_rr import DelayedRR
from sperl.metapolicies.algorithms.mixed_delayed_rr import MixedDelayedRR
from sperl.metapolicies.algorithms.repeated_retraining import RepeatedRetraining

META_POLICIES = [MixedDelayedRR, DelayedRR, RepeatedRetraining]


def get_meta_policy(name: str) -> Type[MetaPolicy]:
    for policy in META_POLICIES:
        if policy.NAME == name:
            return policy
    raise ValueError("{} is not a supported meta policy.".format(name))
