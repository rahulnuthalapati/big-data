//package com.wordcount;


import java.awt.peer.ListPeer;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;
import java.util.regex.Pattern;

import javafx.util.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.yarn.webapp.hamlet2.Hamlet;

public class WordCount
{
    //https://gist.github.com/sebleier/554280
    public static List<String> stopperWords;//new ArrayList<>(Arrays.asList("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"));

    static {
        try {
            stopperWords = Files.readAllLines(new File("stopwords.txt").toPath(), Charset.defaultCharset() );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>
    {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException
        {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens())
            {
                String stringWord = itr.nextToken().toLowerCase(); //For converting upper to lower case (Dog and dog as dog)
//                if(!stopperWords.contains(stringWord)) //For Q3
//                {
                    word.set(stringWord);
                    context.write(word, one);
//                }

            }
        }
    }

    public static class IntSumReducer extends Reducer<Text,IntWritable,Text,IntWritable>
    {
        private IntWritable result = new IntWritable();
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException
        {
            int sum = 0;
            for (IntWritable val : values)
            {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static List<Pair<Text, IntWritable>> allNonPunctWords = new ArrayList<>();

    public static class SortMapper extends Mapper<Object, Object, Text, IntWritable>
    {
        public void map(Object key, Object value, Context context) throws IOException, InterruptedException
        {
            try
            {
                String[] wordsAndCounts = value.toString().split("\t");
                String word = wordsAndCounts[0];
                IntWritable count = new IntWritable(Integer.parseInt(wordsAndCounts[1]));
                Pattern punctPattern = Pattern.compile("\\p{Punct}");
                if(!punctPattern.matcher(word).find())
                {
                    Pair<Text, IntWritable> wordPair = new Pair<>(new Text(word), count);
                    allNonPunctWords.add(wordPair);
                }
            }
            catch (Exception exception)
            {
                StringWriter sw = new StringWriter();
                exception.printStackTrace(new PrintWriter(sw));
                String exceptionAsString = sw.toString();
                context.write(new Text(String.valueOf(exceptionAsString)), new IntWritable(1));
            }
        }

    }

    public static class SortReducer extends Reducer<Text, IntWritable, Text, IntWritable>
    {
        public List<Pair<Text, IntWritable>> allWordsWithoutStopper = new ArrayList<>();
        public List<Pair<Text, IntWritable>> topTwentyFive = new ArrayList<>();
        public List<Pair<Text, IntWritable>> topTwentyFiveWithoutStop = new ArrayList<>();
        //https://gist.github.com/sebleier/554280#file-nltk-s-list-of-english-stopwords

        public void reduce(Context context) throws IOException, InterruptedException
        {
            try
            {
                super.cleanup(context);
            }
            catch (Exception exception)
            {
                StringWriter sw = new StringWriter();
                exception.printStackTrace(new PrintWriter(sw));
                String exceptionAsString = sw.toString();
                context.write(new Text(String.valueOf(exceptionAsString)), new IntWritable(1));
            }
        }

        public void writeToContext(Context context, List<Pair<Text, IntWritable>> list, boolean useTop) throws IOException, InterruptedException
        {
            int top = 0;
            for (Pair<Text, IntWritable> res : list)
            {
                context.write(new Text(res.getKey()), res.getValue());
                top++;
                if(useTop && top >= 25) break;
            }
        }

        //https://hadoop.apache.org/docs/r2.4.1/api/org/apache/hadoop/mapreduce/Mapper.html#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
        @Override
        protected void cleanup(Reducer<Text, IntWritable, Text, IntWritable>.Context context) throws IOException, InterruptedException
        {
            try {
                super.cleanup(context);

                for(Pair<Text, IntWritable> punctWord: allNonPunctWords)
                {
                    if(!stopperWords.contains(String.valueOf(punctWord.getKey())))
                    {
                        allWordsWithoutStopper.add(punctWord);
                    }
                }


                for (Pair<Text, IntWritable> word : allNonPunctWords)
                {
                    for (int i = 0; i < topTwentyFive.size() && i < 25; i++)
                    {
                        Pair<Text, IntWritable> currWord = topTwentyFive.get(i);
                        if ((word.getValue()).compareTo(currWord.getValue()) >= 0)
                        {
                            topTwentyFive.add(i, word);
                            break;
                        }
                    }

                    if (topTwentyFive.isEmpty())
                    {
                        topTwentyFive.add(word);
                    }

                    for (int i = 0; (i < topTwentyFiveWithoutStop.size() && i < 25 && !stopperWords.contains(String.valueOf(word.getKey()))); i++)
                    {
                        Pair<Text, IntWritable> currWord = topTwentyFiveWithoutStop.get(i);
                        if ((word.getValue()).compareTo(currWord.getValue()) >= 0)
                        {
                            topTwentyFiveWithoutStop.add(i, word);
                            break;
                        }
                    }
                    if (topTwentyFiveWithoutStop.isEmpty() && !stopperWords.contains(String.valueOf(word.getKey())))
                    {
                        topTwentyFiveWithoutStop.add(word);
                    }
                }


//                writeToContext(context, allNonPunctWords, false); //All words without punctuation
//                writeToContext(context, topTwentyFive, true); //Top 25 words without punctuation
//                writeToContext(context, allWordsWithoutStopper, false); // All words without stopper and punctuation
                writeToContext(context, topTwentyFiveWithoutStop, true); //Top 25 words without stopper and punctuation
            }
            catch (Exception exception)
            {
                StringWriter sw = new StringWriter();
                exception.printStackTrace(new PrintWriter(sw));
                String exceptionAsString = sw.toString();
                context.write(new Text(String.valueOf(exceptionAsString)), new IntWritable(1));
            }
        }
    }


    public static void main(String[] args) throws Exception
    {
        List<String> stopperWords1 = stopperWords;
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(WordCount.TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        job.waitForCompletion(true);

        Configuration sortConf = new Configuration();
        Job sortJob = Job.getInstance(sortConf, "word sort");
        sortJob.setJarByClass(WordCount.class);
        sortJob.setMapperClass(WordCount.SortMapper.class);
        sortJob.setReducerClass(WordCount.SortReducer.class);
        sortJob.setOutputKeyClass(Text.class);
        sortJob.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(sortJob, new Path(args[1]));
        FileOutputFormat.setOutputPath(sortJob, new Path(args[2]));
        System.exit(sortJob.waitForCompletion(true) ? 0 : 1);
    }
}