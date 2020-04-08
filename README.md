# NSCL-PyTorch
Pytorch implementation for the Neuro-Symbolic Concept Learner (NS-CL) on CLEVRER.


## Dataset preparation
- [x] More carefully-designed features, including normalized boxes,
- [x] Features for collisions

## Reasons for failure cases
- [ ] Fail to track the object
- [ ] Filter Order: 'first' and 'second' 
- [ ] Filter for events: 'collisions'
- [ ] Filter for events: 'stationary' or 'moving' at a moment
- [ ] Filter for objects: 'in' or 'out' at a moment

## Tto do list:
- [x] Temporal operations for 'start', 'end', 'in', 'out', 'before' and 'end'
- [x] New operations for 'collisions' and  'get_col_partner' 
- [x] Temporal feature filtering 'start' and 'end' the box sequences
- [x] More complicated feature representations for box seq
- [x] IoU for collision
- [ ] Diff/RGB for stationary and moving
- [ ] Temporal operation for 'first', 'second' and 'last'
- [ ] Conflict value between default [0, 0, 0, 0] and mask out values [0, 0, 0, 0]

## Temporal reasoning
- [ ] Design neoral operations for temporal reasoning programs
