#! /bin/bash
#for ((exp=1; exp<=5; exp++))
for (( trial=0 ; trial<=20 ; trial++))
do 
	for (( n=1; n<=23000/1000; n++ ))
	do 
  		#n=$(( $exp**2 * 1000 ))
  		n1000=$(( $n * 1000 ))
  		echo $n1000 | tee -a cpu.out
  		./hd $n1000 1000 0 | tee -a cpuTime.out
	done
done 
