# NSCL-PyTorch
Pytorch implementation for the Neuro-Symbolic Concept Learner (NS-CL) on CLEVRER.


## Dataset preparation
- [x] More carefully-designed features, including normalized boxes,
- [x] Features for collisions

## Reasons for failure cases
- [x] Bugs for stationary/moving
- [x] Filter for events: 'stationary' or 'moving' at a moment
- [x] Filter for objects: 'in' or 'out' at a moment
- [ ] Fail to track the object
- [ ] Filter Order: 'first' and 'second' 
- [ ] Filter for events: 'collisions'

## Done List:
- [x] Temporal operations for 'start', 'end', 'in', 'out', 'before' and 'end'
- [x] New operations for 'collisions' and  'get_col_partner' 
- [x] Temporal feature filtering 'start' and 'end' the box sequences
- [x] More complicated feature representations for box seq
- [x] IoU for collision
- [x] Diff for stationary and moving
- [x] Conflict value between default [0, 0, 0, 0] and mask out values [0, 0, 0, 0]
- [x] Filter order programs

## To do list:
More comprehensive features for temporal concepts 
- [x] Gaussion smoothing
- [ ] Multiple choice selection question, program definition V2
- [ ] Skipping bounding boxes
- [ ] New tube proposal generation
- [ ] RGB for stationary and moving
- [ ] Temporal operation for 'first', 'second' and 'last'
- [ ] dynamic program V2 for modeling video graph

## Temporal reasoning
- [ ] Design neoral operations for temporal reasoning programs
