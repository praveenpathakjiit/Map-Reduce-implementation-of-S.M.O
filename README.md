# Map-Reduce-implementation-of-S.M.O
The projet aimed to use the map reduce hadoop framework for the parallelization of Nature inspired algorithm known as Spider monkey optimization

Set up Hadoop environment in your system. Set up master-slave architecture if running on more than one system. Put input file on HDFS using: hadoop put -fs input.txt Then run the MapReduce job using: hadoop jar /usr/hdp/2.6.2.0-205/hadoop-mapreduce/hadoop-streaming.jar -input purchases.txt -output op/op1 -mapper mapper.py -file mapper1.py -reducer reducer.py -file reducer1.py Then see the output through the output file made on HDFS.
