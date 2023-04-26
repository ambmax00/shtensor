#! /bin/bash

image_name="shtensor"

nodes=1
procs=1

options=$(getopt -o '' --long nodes:,procs: -- "$@")
eval set -- "$options"
while true; do
  case $1 in
    --nodes )
      nodes=$2
      shift 2
      ;;
    --procs )
      procs=$2
      shift 2
      ;;
    -- )
      shift 
      break
      ;;
    * )
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
done

directory=$1

if [ -z "$directory" ]; then
  echo "Usage: ./launch.sh --nodes=n --procs=p TRUNK_DIR"
  exit 0
fi

echo "Launching $nodes containers with $procs processes each"

echo "Attaching to directory $directory"

if [ `docker images | grep $image_name | wc -l` -eq 0 ]; then
  echo "Could not find docker image shtensor"
  exit 1
fi

# generate node names
node_list=()

for i in $(seq 1 "$nodes"); do
  container_name=`echo host${i}`
  node_list+=("$container_name")
done

declare -A node_ips

for node in ${node_list[@]}; do
  echo "Launching $node"
  
  docker run --hostname $node -v $directory:/home/dev/trunk --name $node \
    -t -d -i --rm $image_name /usr/sbin/sshd -D > /dev/null

  ip_address=`docker inspect \
              -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' \
              ${node}`

  node_ips[$node]=$ip_address
done

for node in ${node_list[@]}; do
  echo "Generating hostfiles for $node"
  for nnode in ${node_list[@]}; do
    [ "$node" = "$nnode" ] && continue 
    docker exec $node bash -c "echo ${node_ips[$nnode]} $nnode >> /etc/hosts"
    done
done

# generate machine file
machine_file=$(mktemp)

for node in ${node_list[@]}; do
  echo "$node slots=$procs" >> $machine_file
done

# prepare entry node
docker cp $machine_file ${node_list[0]}:/etc/openmpi/openmpi-default-hostfile 

# do stuff 
docker exec -it --user dev ${node_list[0]} /bin/bash

for node in ${node_list[@]}; do
  echo "Removing $node"
  docker stop $node
done

