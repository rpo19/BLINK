# Patch AIDA dataset creation JAR

## Download the AIDA dataset
- Enter the `aida-patch` folder.
- Download the `aida-yago2-dataset.zip` as explained in the section `AIDA CoNLL-YAGO Dataset` from [here](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads). Then extract it.
- Download the CoNLL data as described in `aida-yago2-dataset/README_CoNLL_2003.TXT`.

## The modification
The modification consists in patching the class `mpi/truths/Convert.class` inside `aida-yago2-dataset.jar` so that it considers NER types too. This is the difference from the original source file:
```
--- original/Convert.java
+++ patch/Convert.java
@@ -75,7 +75,7 @@
                         char startDef = 'B';
                         while (i < iterateTo) {
                             token = tokens.getToken(i);
-                            writer.write(String.valueOf(token.getOriginal()) + "\t" + startDef + "\t" + text + "\t" + mention.getGroundTruthResult() + "\n");
+                            writer.write(String.valueOf(token.getOriginal()) + "\t" + startDef + "\t" + text + "\t" + mention.getGroundTruthResult() + "\t" + token.getNE() + "\n");
                             if (start) {
                                 startDef = 'I';
                                 start = false;
```
## How to patch
### Get the patch
Get the patched file (`mpi/truths/Convert.java`) respecting the directory structure:
```
> find mpi/
mpi/
mpi/truths
mpi/truths/Convert.java
```


### Compile the patch:
```
javac -classpath aida-yago2-dataset/aida-yago2-dataset.jar mpi/truths/Convert.java
```
As a results the file `Convert.class` should have been created:
```
> find mpi/
mpi/
mpi/truths
mpi/truths/Convert.class
mpi/truths/Convert.java
```
### Inject the patched class into the JAR
```
jar uf aida-yago2-dataset/aida-yago2-dataset.jar mpi/truths/Convert.class
```

### Proceed with the default instructions
Now proceed with the default AIDA instructions on how to create the AIDA dataset starting from CoNLL2013. They are available in the README files previously extracted from `aida-yago2-dataset.zip`.
It should be as simple as running:
```
cd aida-yago2-dataset
java -jar aida-yago2-dataset.jar
# enter the folder containing CoNLL data
```
Then the dataset with NER types should be at `aida-yago2-dataset/AIDA-YAGO2-dataset.tsv`