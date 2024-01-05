import argparse
import os
from datetime import datetime

import boto3
import sagemaker
# from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import FileSystemInput


def get_job_name(base):
    now = datetime.now()
    now_ms_str = f'{now.microsecond // 1000:03d}'
    date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
    job_name = '_'.join([base, date_str])
    return job_name


def launch(args):

    if args.wandb_api_key is None:
        wandb_api_key = os.environ.get('WANDB_API_KEY', None)
        assert wandb_api_key is not None, 'Please provide wandb api key either via --wandb-api-key or env variable WANDB_API_KEY.'
        args.wandb_api_key = wandb_api_key

    if args.local:
        assert args.instance_count == 1, f'Local mode requires 1 instance, now get {args.instance_count}.'
        assert args.input_source not in {'lustre'}
        args.sagemaker_session = sagemaker.LocalSession()
    else:
        assert args.input_source not in {'local'}    
        args.sagemaker_session = sagemaker.Session()

    if args.input_source == 'local':
        input_mode = 'File'
        # training_inputs = {"calvin_data_processed":'file:///home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned'}
        training_inputs = {"calvin_data_processed":'file:///home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned',
                           "something_something_processed":'file:///home/kylehatch/Desktop/hidql/data/something_something_processed'}
    elif args.input_source == 'lustre':
        input_mode = 'File'
        train_fs = FileSystemInput(
            file_system_id='fs-02831553b25f26b1c', ###TODO
            file_system_type='FSxLustre',
            directory_path='/onhztbev', ###TODO
            file_system_access_mode='ro'
        )
    elif args.input_source == 's3':
        input_mode = 'FastFile'
        # train_fs = 's3://tri-ml-datasets/scratch/dianchen/datasets/' ###TODO
        training_inputs = {"calvin_data_processed":'s3://susie-data/calvin_data_processed/language_conditioned/',
                           "something_something_processed":"s3://susie-data/something_something_processed/"}
    else:
        raise ValueError(f'Invalid input source {args.input_source}')

    role = 'arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess'
    role_name = role.split(['/'][-1])

    session = boto3.session.Session()
    region = session.region_name

    config = os.path.join('/opt/ml/code/', args.config)
    hyperparameters = {
        'config': config
    }




    subnets = [
        # 'subnet-07bf42d7c9cb929e4',
        # 'subnet-0f72615fd9bd3c717', 
        # 'subnet-0a29e4f1a47443e28', 
        # 'subnet-06e0db77592be2b36',

        'subnet-05f1115c7d6ccbd07',
        'subnet-03c7c7be28e923670',
        'subnet-0a29e4f1a47443e28',
        'subnet-06e0db77592be2b36',
        'subnet-0dd3f8c4ce7e0ae4c',
        'subnet-02a6ddd2a60a8e048',
        'subnet-060ad40beeb7f24b4',
        'subnet-036abdaead9798455',
        'subnet-07ada213d5ef651bb',
        'subnet-0e260ba29726b9fbb',
        'subnet-08468a58663b2b173',
        'subnet-0ecead4af60b3306f',
        'subnet-09b3b182287e9aa29',
        'subnet-07bf42d7c9cb929e4',
        'subnet-0f72615fd9bd3c717',
        'subnet-0578590f6bd9a5dde',
        # 'subnet-03550978b510a6d55',
        # 'subnet-0449e12487555a62a',
        # 'subnet-0a930733cdb95ffc9',
        # 'subnet-07d4dd2bc160d9df6',
        # 'subnet-016ef5d3e0df2ab9d',
        # 'subnet-0a2097ea05c20b45e',
        # 'subnet-0310cca5c76f96899',
        # 'subnet-05a638cfbfae73305',
        # 'subnet-03853f6eef9dbd13f',
    ]


    security_group_ids = [
        'sg-0afb9fb0e79a54061', 
        'sg-0333993fea1aeb948', 
        'sg-0c4b828f4023a04cc',
    ]



    job_name = get_job_name(args.base_job_name)

    if args.local:
        image_uri = f'{args.base_job_name}:latest' 
    else:
        image_uri = f'124224456861.dkr.ecr.us-east-1.amazonaws.com/{args.base_job_name}:latest'
    
    output_path = os.path.join(f's3://tri-ml-sandbox-16011-us-east-1-datasets/sagemaker/{args.user}/susie/', job_name)

    checkpoint_s3_uri = None if args.local else output_path
    checkpoint_local_path = None if args.local else '/opt/ml/checkpoints'
    code_location = output_path

    base_job_name = args.base_job_name
    instance_count = args.instance_count
    entry_point = args.entry_point
    sagemaker_session = args.sagemaker_session

    instance_type = 'local_gpu' if args.local else args.instance_type 
    keep_alive_period_in_seconds = 0
    max_run = 60 * 60 * 24 * 5

    environment = {
        'PYTHONPATH': '/opt/ml/code:/opt/ml/code/externals/datasets:/opt/ml/code/external/dlimp', 
        'WANDB_API_KEY': args.wandb_api_key,
        'EXP_DESCRIPTION': args.exp_description,

        # "CUDA_VISIBLE_DEVICES":"1",
        # "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
    }

    distribution = {
        'smdistributed': {
            'dataparallel': {
                    'enabled': True,
            },
        },
    }

    # inputs = {
    #     'training': train_fs,
    # }

    print()
    print()
    print('#############################################################')
    print(f'SageMaker Execution Role:       {role}')
    print(f'The name of the Execution role: {role_name[-1]}')
    print(f'AWS region:                     {region}')
    print(f'Entry point:                    {entry_point}')
    print(f'Image uri:                      {image_uri}')
    print(f'Job name:                       {job_name}')
    print(f'Configuration file:             {config}')
    print(f'Instance count:                 {instance_count}')
    print(f'Input mode:                     {input_mode}')
    print('#############################################################')
    print()
    print()

    # estimator = PyTorch(
    #     base_job_name=base_job_name,
    #     entry_point=entry_point,
    #     hyperparameters=hyperparameters,
    #     role=role,
    #     image_uri=image_uri,
    #     instance_count=instance_count,
    #     instance_type=instance_type,
    #     environment=environment,
    #     sagemaker_session=sagemaker_session,
    #     subnets=subnets,
    #     security_group_ids=security_group_ids,
    #     keep_alive_period_in_seconds=keep_alive_period_in_seconds,
    #     max_run=max_run,
    #     input_mode=input_mode,
    #     job_name=job_name,
    #     output_path=output_path,
    #     checkpoint_s3_uri=checkpoint_s3_uri,
    #     checkpoint_local_path=checkpoint_local_path,
    #     code_location=code_location,
    #     distribution=distribution,
    # )
    # estimator.fit(inputs=inputs)

    if args.enable_ddp:
        estimator = TensorFlow(
            base_job_name=base_job_name,
            entry_point=entry_point,
            hyperparameters=hyperparameters,
            role=role,
            image_uri=image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            environment=environment,
            sagemaker_session=sagemaker_session,
            subnets=subnets,
            security_group_ids=security_group_ids,
            keep_alive_period_in_seconds=keep_alive_period_in_seconds,
            max_run=max_run,
            input_mode=input_mode,
            job_name=job_name,
            # output_path=output_path,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path=checkpoint_local_path,
            code_location=code_location,
            distribution=distribution,
        )
    else:
        # distribution_fake = {
        #     'smdistributed': {
        #         'dataparallel': {
        #                 'enabled': False,
        #         },
        #     },
        # }

        estimator = TensorFlow(
        base_job_name=base_job_name,
        entry_point=entry_point,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        environment=environment,
        sagemaker_session=sagemaker_session,
        subnets=subnets,
        security_group_ids=security_group_ids,
        keep_alive_period_in_seconds=keep_alive_period_in_seconds,
        max_run=max_run,
        input_mode=input_mode,
        job_name=job_name,
        # output_path=output_path,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        code_location=code_location,
        # distribution=distribution,
        # distribution=distribution_fake,
    )
    estimator.fit(inputs=training_inputs)

    # estimator = TensorFlow(
    #     role=role,
    #     instance_count=1,
    #     base_job_name="jax",
    #     framework_version="2.10",
    #     py_version="py39",
    #     source_dir="training_scripts",
    #     entry_point="train_jax.py",
    #     instance_type="ml.p3.2xlarge",
    #     hyperparameters={"num_epochs": 3},
    # )
    # estimator.fit(logs=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--base-job-name', type=str, required=True)
    parser.add_argument('--user', type=str, required=True, help='supported users under the IT-predefined bucket.')
    parser.add_argument('--wandb-api-key', type=str, default=None)
    parser.add_argument('--input-source', choices=['s3', 'lustre', 'local'], default='lustre')
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--entry_point', type=str, default='scripts/train.py'),
    parser.add_argument('--enable_ddp', action='store_true', default=False)
    parser.add_argument('--exp_description', type=str, default=None),
    parser.add_argument('--instance_type', type=str, default="ml.p4de.24xlarge"),
    args = parser.parse_args()

    launch(args)

"""
# Local debug, no SMDDP
./update_docker.sh
python3 -u sagemaker_launch.py \
--entry_point train.py \
--user kylehatch \
--local \
--input-source local \
--config configs/base.py:sagemaker_local_debug \
--base-job-name susie \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--exp_description local


# Remote no SMDDP debug
python3 -u sagemaker_launch.py \
--entry_point train.py \
--user kylehatch \
--input-source s3 \
--config configs/base.py:sagemaker_debug \
--base-job-name susie \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--exp_description remote \
--instance_type ml.p4de.24xlarge

# Remote no SMDDP
python3 -u sagemaker_launch.py \
--entry_point train.py \
--user kylehatch \
--input-source s3 \
--config configs/base.py:sagemaker \
--base-job-name susie \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--exp_description remote400smth \
--instance_type ml.p4de.24xlarge

python3 -u sagemaker_launch.py \
--entry_point train.py \
--user kylehatch \
--input-source s3 \
--config configs/base.py:sagemaker \
--base-job-name susie \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--exp_description remote128smth \
--instance_type ml.p4de.24xlarge




Try EC2 
- P4DE instance 


Try:
- Try preallocate false 
- Pytorch and jax side by side, check size of the model 
- See where exzactly in the code it is OOMing 
- manually initializing the cluster 
    - Myabe don't use sagemaker ddp thing, but have a third script to do this manually 
        - scratch tests first 

- Testing GPU OOM behavior on sagemaker local 
    - "CUDA_VISIBLE_DEVICES":"1" and "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
        - OOMs down to batch size 32 
        - Works on batch size 16 (both w and w/o sagemaker)
        - "XLA_PYTHON_CLIENT_PREALLOCATE":"false" doesn't seem to make a difference 
    - Works with batch size 96,99 with 3 gpus 
    - OOMs with batch size 102,108,120,192 with 3 gpus

- check max batch pytorch size 
    - pytorch can do 64,96,116,123 per gpu, but OOMs on 124,128 per gpu 

24 Dec 23
Current issue:
- On Sagemaker, OOMs with large batch size with a single process
    - Even though it has 8 gpus, running it in a single process still makes it OOM
    - both 1024 and 512 
- Works fine on with a single process with a small batch size of XXX
- the train.py code is set up for multiprocessing
    - not sure how to launch, waiting to hear from Kevin 
    - launching with sagemaker mutliprocessing, the jax script doesn't recognize its being run as a multiprocess thing
        - each script just thinks its being run independently 


Tried
- just set enabled=False
    - Just does the same thing as not passing in the distribution dict to the estimator (ie: no distributed)
"""