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
- [x] Gaussion smoothing
- [x] Multiple choice selection question, program definition V2
- [x] Skipping bounding boxes
- [x] New tube proposal generation
- [x] Temporal operation for 'first', 'second' and 'last'
- [x] dynamic program V2 for modeling video graph
- [x] Learning dynamic models in the latent space
- [x] Finding out why it fails, using deeper model
- [x] QA accuracy per question

## To do list:
More comprehensive features for temporal concepts 
- [ ] Models with learnt tube proposals
- [ ] Models with learnt programs
- [ ] Threshold for collision performance
- [ ] More supervision info from QA
### Details for learning dynamic modeles in the latent space
Step1, extract visual features for each target object and their collision features
Step2, using a dynamic models to learn corresponding features
Step3, Joint optimizing both the latent features and the dynamic nscl models

## Temporal reasoning
- [ ] Design neoral operations for temporal reasoning programs

## Minor to do list:
- [ ] RGB for stationary and moving
