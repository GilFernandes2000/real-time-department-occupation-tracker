#!/bin/bash
if [ -z "$1" ] || [ -z "$2" ]; then
	echo Not enough input arguments provided
	echo Please specify interface name and IP/NETMASK
	echo Example: ./create_batman_interface wlan0 10.1.1.1/24

else
	sudo ip link set $1 down
	sudo iw $1 set type ibss
	sudo ifconfig $1 mtu 1500
	sudo iwconfig $1 channel 6 # channel 6 was foun to have the least traffic in the IT2 network
	sudo ip link set $1 up
	sudo iw $1 ibss join atlas 2437 

	sudo modprobe batman-adv
	sudo batctl if add $1
	sudo ip link set up dev $1
	sudo ip link set up dev bat0
	sudo ifconfig bat0 $2
fi
