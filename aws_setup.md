# Setting up instance for training on AWS

Commands for creating EC2 instances and docker images for training and evaluating models for Student Engagement Project.

### 1. Pre-reqs

- Check default VPC

```
aws ec2 describe-vpcs
```

- Create Security Group - allow ports 22, 6006 for tensorboard and 8000 for jupyter

```
$ aws ec2 create-security-group --group-name w251-project --description "Security Group for w251 project" --vpc-id vpc-bc3d79c4  

$ aws ec2 authorize-security-group-ingress --group-id  sg-06e553796ccf840c9 --protocol tcp --port 22 --cidr 0.0.0.0/0

$ aws ec2 authorize-security-group-ingress --group-id  sg-06e553796ccf840c9 --protocol tcp --port 6006 --cidr 0.0.0.0/0

$ aws ec2 authorize-security-group-ingress --group-id  sg-06e553796ccf840c9 --protocol tcp --port 8000 --cidr 0.0.0.0/0
```

- Create EFS Volume for persistence  

  A common EFS volume is useful for persisting data between terminated AWS instances, or different instances

  - Use Amazon Web Interface  
  - Make note of IP

- Create EBS Volume for local storage as there isn't enough on the server by default
aws ec2 create-volume --volume-type gp2  --size 100 --availability-zone us-west-2a


### 3. Create instance with GPU - for training models

- Create Amazon Instance (not spot as need to start and stop)  
Use the following AMI (US West 2) as it contains CUDA libraries
```
aws ec2 run-instances --image-id  ami-0bc87a16c757a7f07 --security-group-ids sg-06e553796ccf840c9  --instance-type p3.2xlarge --key-name aws-MIDS  
```

- NB can also create compute intensive where GPU is not required, or even t2.micro for mounting and moving files, just change the ami and instance type, e.g. for compute:
-- `--image-id ami-0013ea6a76d3b8874`
-- `--instance-type c4.8xlarge`

- Attache EBS volume
aws ec2 attach-volume --volume-id vol-0b0f61091e0fb8cef --instance-id i-06205dd0fa3b11ba5 --device /dev/sdf

- Get Public IP
```
aws ec2 describe-instances --query "Reservations[*].Instances[*].PublicIpAddress"  --output=text
```

- ssh
```
ssh -i ~/.ssh/aws-MIDS.pem ubuntu@ec2-34-221-33-165.us-west-2.compute.amazonaws.com
```

-- git clone the repo
```
git clone
```

- create EBS file system, make sure it is in home dir so docker can see it
```
lsblk
sudo file -s /dev/xvdf
cd ~/w251_ChrisSexton/Project/
mkdir data
sudo mount /dev/xvdf /home/ubuntu/w251_ChrisSexton/Project/data
cd data
sudo chmod go+rw .
```



- Mount efs directory
```
mkdir efs
sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport 172.31.32.223:/ efs
cd efs
sudo chmod go+rw .
```

- Copy data to EBS volume
```
mkdir ~/w251_ChrisSexton/Project/data/DAiSEE
cp -R ~/efs/DAiSEE/DataSet ~/w251_ChrisSexton/Project/data/DAiSEE
```

- Install Docker
```
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common  
```
```       
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```
```
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```
```
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
```

- Get docker images and build container
```
cd
docker pull tensorflow/tensorflow:latest-gpu-jupyter
sudo docker run  --gpus all   -ti -p 8888:8888 -v /home/ubuntu/w251_ChrisSexton/Project:/Project tensorflow/tensorflow:latest-gpu-jupyter bash
# install packages
apt-get update
apt-get install -y ffmpeg
pip3 install keras
pip3 install efficientnet
apt-get install -y python3-sklearn python3-sklearn-lib
pip3 install pandas
apt-get install -y python3-opencv
pip3 install opencv-contrib-python
pip3 install seaborn
```

- save docker container as new image
```
docker commit <container name> csexton/tensorflow-opwncv:version1
```

- next time just run the container
```
sudo docker run  --gpus all  -ti -p 8888:8888 -v /home/ubuntu/w251_ChrisSexton/Project:/Project csexton/tensorflow-opwncv:version1 bash
```

- start the notebook server and connect (locally)
```
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root &
http://52.42.239.61:8888/?token=0ed3817f22e5ee03dad246ae5cf787215a231220aecf96ae
```

# setup Broker

- create an instance
```
aws ec2 run-instances --image-id  ami-0013ea6a76d3b8874 --security-group-ids sg-06e553796ccf840c9  --instance-type t2.micro --key-name aws-MIDS  
```

- get the public dns name
```
aws ec2 describe-instances | grep PublicDnsName
```

- copy docker files to instances
```
scp -i ~/.ssh/aws-MIDS.pem Dockerfile.broker ubuntu@ec2-54-69-37-72.us-west-2.compute.amazonaws.com:/home/ubuntu/  
scp -i ~/.ssh/aws-MIDS.pem Dockerfile.processor ubuntu@ec2-54-69-37-72.us-west-2.compute.amazonaws.com:/home/ubuntu/
scp -i ~/.ssh/aws-MIDS.pem processor.py ubuntu@ec2-54-69-37-72.us-west-2.compute.amazonaws.com:/home/ubuntu/
```

- ssh
```
ssh -i ~/.ssh/aws-MIDS.pem ubuntu@ec2-54-69-37-72.us-west-2.compute.amazonaws.com
```

- install docker and depencies
```
sudo su
apt-get update
apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
apt-get install -y docker-ce
```

- setup network bridge and build the docker images
```
sudo docker network create --driver bridge project
sudo docker build -t aws_broker -f Dockerfile.broker .
sudo docker build -t aws_processor -f Dockerfile.processor .
```

-Start individual docker instances

```
sudo docker run --name aws_broker --network project -p 1883:1883 -ti aws_broker mosquitto
sudo docker run --name aws_processor --privileged --network project -ti aws_processor bash
```

- Inside aws_processor container - cannot map network drive from dockerfile so must do manually

```
s3fs s3-engagement -o use_cache=/tmp -o allow_other -o uid=1000 -o mp_umask=002 -o multireq_max=5 /usr/src/app/engagement/s3
python /usr/src/app/engagement/processor.py
```
