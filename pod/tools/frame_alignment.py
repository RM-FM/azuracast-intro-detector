"""Align 10 ms classifier/energy frames without stretching their timeline."""
def align_frames(scores, energy, target, eof=False):
    if not scores or not energy: raise ValueError('Empty classifier/energy output')
    # PANNs pooling/interpolation can omit up to one 32-frame block at the tail.
    score_gap=max(0,target-len(scores)); energy_gap=max(0,target-len(energy))
    if score_gap>32 or energy_gap>(25 if eof else 2):
        raise ValueError(f'Analysis chunk duration mismatch: target={target}, classifier={len(scores)}, energy={len(energy)}, eof={eof}')
    # Repeat terminal predictions; never stretch/rescale all timestamps.
    # Extend energy conservatively, rather than declaring an unobserved tail silent.
    return ((scores+[scores[-1]]*score_gap)[:target],
            (energy+[energy[-1]]*energy_gap)[:target])
