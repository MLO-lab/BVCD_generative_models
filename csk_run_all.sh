#!/bin/bash

model='opt-13b'
run_id='1rj7gglg'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
model='opt-6.7b'
run_id='1fg1y4wi'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
model='opt-2.7b'
run_id='2e9bcge4'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
model='opt-1.3b'
run_id='q7jwqjuc'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
model='opt-350m'
run_id='r6bcg0j0'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
model='opt-125m'
run_id='34webe3u'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id

model='opt-13b'
run_id='349oybw1'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
model='opt-6.7b'
run_id='38nlrhxa'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
model='opt-2.7b'
run_id='3aws5stu'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
model='opt-1.3b'
run_id='2yotsedc'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
model='opt-350m'
run_id='386igsi5'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
model='opt-125m'
run_id='vqsivevv'
python get_ncsk_entropy.py --generation_model=$model --run_id=$run_id
