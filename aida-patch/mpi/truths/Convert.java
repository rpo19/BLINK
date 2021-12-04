package mpi.truths;

import java.io.IOException;
import mpi.truths.data.Mentions;
import mpi.truths.data.Mention;
import mpi.truths.data.AdvToken;
import java.util.Iterator;
import mpi.truths.data.AdvTokens;
import java.io.Writer;
import java.io.BufferedWriter;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.FileOutputStream;
import mpi.truths.data.Data;
import java.io.File;

public class Convert
{
    private String publicFile;
    private static final String finalName = "AIDA-YAGO2-dataset.tsv";
    private File result;
    private String conllPath;
    private Data data;
    
    public Convert() {
        this.publicFile = "./data/public-aida.tsv";
        this.result = null;
        this.conllPath = "./data/conll";
        this.data = null;
    }
    
    public Convert(final String conllPath, final String publicFile) {
        this.publicFile = "./data/public-aida.tsv";
        this.result = null;
        this.conllPath = "./data/conll";
        this.data = null;
        this.conllPath = conllPath;
        this.publicFile = publicFile;
    }
    
    public void start() {
        (this.data = new Data()).init(this.conllPath, this.publicFile);
        try {
            final File f = new File(this.publicFile);
            this.result = new File(f.getParentFile(), "AIDA-YAGO2-dataset.tsv");
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        this.writeData();
    }
    
    private void writeData() {
        try {
            final FileOutputStream fos = new FileOutputStream(this.result);
            final BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fos, "UTF-8"));
            for (final String docId : this.data.getDocIds()) {
                writer.write("-DOCSTART- (" + docId + ")\n");
                final AdvTokens tokens = this.data.getDocuments().get(docId);
                AdvToken prevToken = null;
                for (int i = 0; i < tokens.size(); ++i) {
                    AdvToken token = tokens.getToken(i);
                    if (prevToken != null && token.getSentence() > prevToken.getSentence()) {
                        writer.write("\n");
                    }
                    final Mention mention = this.getTargetMention(token.getId(), docId);
                    if (mention != null) {
                        final String text = mention.getMention();
                        final int iterateTo = mention.getEndToken() + 1;
                        boolean start = true;
                        char startDef = 'B';
                        while (i < iterateTo) {
                            token = tokens.getToken(i);
                            writer.write(String.valueOf(token.getOriginal()) + "\t" + startDef + "\t" + text + "\t" + mention.getGroundTruthResult() + "\t" + token.getNE() + "\n");
                            if (start) {
                                startDef = 'I';
                                start = false;
                            }
                            ++i;
                        }
                        --i;
                    }
                    else {
                        writer.write(String.valueOf(token.getOriginal()) + "\n");
                    }
                    prevToken = token;
                }
                writer.write("\n");
            }
            writer.flush();
            writer.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    private Mention getTargetMention(final int id, final String docId) {
        final Mentions mentions = this.data.getDocMentions().get(docId);
        return mentions.getMentionForTokenId(id);
    }
    
    public static void main(final String[] args) {
        final String publicFile = "AIDA-YAGO2-annotations.tsv";
        String conllPath = null;
        try {
            while (conllPath == null) {
                System.out.flush();
                System.out.println("Specify the directory containing the CoNLL 2003 files 'eng.train', 'eng.testa', and 'eng.testb':");
                conllPath = read(true);
                final File dir = new File(conllPath);
                final File train = new File(dir, "eng.train");
                final File testa = new File(dir, "eng.testa");
                final File testb = new File(dir, "eng.testb");
                if (!train.exists() || !testa.exists() || !testb.exists()) {
                    System.out.println("The specified directory did not contain the required files!\n");
                    conllPath = null;
                }
            }
            final File annotations = new File(publicFile);
            if (!annotations.exists()) {
                System.out.println("'AIDA-YAGO2-annotations.tsv' could not be found, make sure it is in the directory where you call the jar!");
                System.exit(10);
            }
        }
        catch (Exception e) {
            System.out.println("There was a Problem:");
            e.printStackTrace();
            return;
        }
        if (conllPath != null && publicFile != null) {
            final Convert convert = new Convert(conllPath, publicFile);
            convert.start();
            System.out.println("The dataset was created: AIDA-YAGO2-dataset.tsv");
        }
    }
    
    public static String read(final boolean directory) throws IOException {
        int c = 1;
        final StringBuffer sb = new StringBuffer(20);
        while ((c = (char)System.in.read()) >= 0 && c != 10) {
            sb.append((char)c);
        }
        final File f = new File(sb.toString().trim());
        if (!f.exists()) {
            System.out.println("WARNING: The file or folder " + sb.toString() + " does not exist.");
            return null;
        }
        if (directory && !f.isDirectory()) {
            System.out.println("WARNING: " + sb.toString() + " is not a directory.");
            return null;
        }
        if (!directory && !f.isFile()) {
            System.out.println("WARNING: " + sb.toString() + " is not a file.");
            return null;
        }
        return sb.toString();
    }
}
