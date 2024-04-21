package rw.nzeyi.example.android.pytorch.asr;

import java.util.*;

public class CTCBeamSearch {

    static final int SIL_ID = 5;
    static final int BLANK_ID = 6;

    static String[] VOCAB_TOKENS = new String[]{"<pad>", "<unk>", "<mask>", "<s>", "</s>", "|", "~", "i", "u", "o", "a", "e", "b", "c", "d", "f", "g", "h", "j", "k", "m", "n", "p", "r", "l", "s", "t", "v", "y", "w", "z", "bw", "by", "cw", "cy", "dw", "fw", "gw", "hw", "kw", "jw", "jy", "ny", "mw", "my", "nw", "pw", "py", "rw", "ry", "sw", "sy", "tw", "ty", "vw", "vy", "zw", "pf", "ts", "sh", "shy", "mp", "mb", "mf", "mv", "nc", "nj", "nk", "ng", "nt", "nd", "ns", "nz", "nny", "nyw", "byw", "ryw", "shw", "tsw", "pfy", "mbw", "mby", "mfw", "mpw", "mpy", "mvw", "mvy", "myw", "ncw", "ncy", "nsh", "ndw", "ndy", "njw", "njy", "nkw", "ngw", "nsw", "nsy", "ntw", "nty", "nzw", "shyw", "mbyw", "mvyw", "nshy", "nshw", "nshyw", "bg", "pfw", "pfyw", "vyw", "njyw", "x", "q", ",", ".", "?", "!", "-", ":", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "\'"};
    static Map<Integer, String> KINSPEAK_VOCAB_IDX = new HashMap<>();

    static {
        for (int i = 0; i < VOCAB_TOKENS.length; i++) {
            KINSPEAK_VOCAB_IDX.put(i, VOCAB_TOKENS[i]);
        }
    }

    static String id_sequence_to_text(List<Integer> seq) {
        StringBuilder bld = new StringBuilder();
        for (Integer id : seq) {
            if ((id != BLANK_ID) && (id > 4)) {
                if (id == SIL_ID) {
                    bld.append(" ");
                } else {
                    bld.append(KINSPEAK_VOCAB_IDX.get(id));
                }
            }
        }
        return String.join(" ", bld.toString().split(" "));
    }

    public static double log(double x) {
        return Math.log(x + 1e-32);
    }

    public static double logaddexp(double x0, double x1) {
        double c = Math.max(x0, x1);
        return c + Math.log(Math.exp(x0 - c) + Math.exp(x1 - c));
    }

    static class BeamEntry implements Comparable {
        double pr_total = log(0);
        double pr_non_blank = log(0);
        double pr_blank = log(0);
        double pr_text = log(1);
        boolean lm_applied = false;
        List<Integer> labeling = new ArrayList<>();

        BeamEntry() {
        }

        @Override
        public int compareTo(Object o) {
            if (o instanceof BeamEntry) {
                BeamEntry other = (BeamEntry) o;
                return Double.compare(other.pr_total + other.pr_text, this.pr_total + this.pr_text);
            }
            return 0;
        }
    }

    static String ctc_beam_search(List<float[]> log_probs_table,
                                  int beam_width) {
        int max_T = log_probs_table.size();
        // int max_C = log_probs_table.get(0).length;
        // initialise beam state
        BeamEntry en = new BeamEntry();
        en.pr_blank = log(1);
        en.pr_total = log(1);
        List<BeamEntry> last_list = Collections.singletonList(en);
        // go over all time-steps
        for (int t = 0; t < max_T; t++) {
            List<BeamEntry> current_list = new ArrayList<>();
            Collections.sort(last_list);
            // go over best beams
            for (int b = 0; (b < beam_width) && (b < last_list.size()); b++) {
                BeamEntry entry = last_list.get(b);
                // double pr_total = entry.pr_total;
                List<Integer> labeling = entry.labeling;
                // probability of paths ending with a non-blank
                double pr_non_blank = log(0);
                if (!labeling.isEmpty()) {
                    // probability of paths with repeated last char at the end
                    pr_non_blank = entry.pr_non_blank + log_probs_table.get(t)[labeling.get(labeling.size() - 1)];
                }
                // probability of paths ending with a blank
                double pr_blank = entry.pr_total + log_probs_table.get(t)[BLANK_ID];

                // fill in data for current beam
                BeamEntry new_entry = new BeamEntry();
                new_entry.labeling = new ArrayList<>(labeling);
                new_entry.pr_non_blank = logaddexp(new_entry.pr_non_blank, pr_non_blank);
                new_entry.pr_blank = logaddexp(new_entry.pr_blank, pr_blank);
                new_entry.pr_total = logaddexp(new_entry.pr_total, logaddexp(pr_blank, pr_non_blank));
                new_entry.pr_text = entry.pr_text;
                new_entry.lm_applied = true;  // LM already applied at previous time-step for this beam-labeling
                current_list.add(new_entry);
                // extend current beam-labeling
                // boolean breaking = false;
                for (Map.Entry<Integer, String> ment : KINSPEAK_VOCAB_IDX.entrySet()) {
                    int c = ment.getKey();
                    List<Integer> new_labeling = new ArrayList<>(labeling);
                    new_labeling.add(c);
                    // if new labeling contains duplicate char at the end, only consider paths ending with a blank
                    pr_non_blank = entry.pr_total + log_probs_table.get(t)[c];
                    if (!labeling.isEmpty()) {
                        if (labeling.get(labeling.size() - 1) == c) {
                            pr_non_blank = entry.pr_blank + log_probs_table.get(t)[c];
                        }
                    }
                    // fill in data
                    BeamEntry additional_entry = new BeamEntry();
                    additional_entry.labeling = new ArrayList<>(new_labeling);
                    additional_entry.pr_non_blank = logaddexp(additional_entry.pr_non_blank, pr_non_blank);
                    additional_entry.pr_total = logaddexp(additional_entry.pr_total, pr_non_blank);
                    current_list.add(additional_entry);
                }
            }
            // set new beam state
            last_list = current_list;
        }
        // normalise LM scores according to beam-labeling-length
        for (BeamEntry entry : last_list) {
            int labeling_len = entry.labeling.size();
            entry.pr_text = (1.0 / ((labeling_len > 0) ? labeling_len : 1.0)) * entry.pr_text;
        }
        // sort by probability
        Collections.sort(last_list);
        return id_sequence_to_text(last_list.get(0).labeling);
    }

    public static class CTCBeamDecoder {
        private List<BeamEntry> last_list;
        private final int beam_width;
        public CTCBeamDecoder(int beam_width) {
            BeamEntry en = new BeamEntry();
            en.pr_blank = log(1);
            en.pr_total = log(1);
            this.last_list = Collections.singletonList(en);
            this.beam_width=beam_width;
        }

        public String onNewInput(float[] log_probs) {
            List<BeamEntry> current_list = new ArrayList<>();
            Collections.sort(this.last_list);
            // go over best beams
            for (int b = 0; (b < this.beam_width) && (b < this.last_list.size()); b++) {
                BeamEntry entry = this.last_list.get(b);
                // double pr_total = entry.pr_total;
                List<Integer> labeling = entry.labeling;
                // probability of paths ending with a non-blank
                double pr_non_blank = log(0);
                if (!labeling.isEmpty()) {
                    // probability of paths with repeated last char at the end
                    pr_non_blank = entry.pr_non_blank + log_probs[labeling.get(labeling.size() - 1)];
                }
                // probability of paths ending with a blank
                double pr_blank = entry.pr_total + log_probs[BLANK_ID];

                // fill in data for current beam
                BeamEntry new_entry = new BeamEntry();
                new_entry.labeling = new ArrayList<>(labeling);
                new_entry.pr_non_blank = logaddexp(new_entry.pr_non_blank, pr_non_blank);
                new_entry.pr_blank = logaddexp(new_entry.pr_blank, pr_blank);
                new_entry.pr_total = logaddexp(new_entry.pr_total, logaddexp(pr_blank, pr_non_blank));
                new_entry.pr_text = entry.pr_text;
                new_entry.lm_applied = true;  // LM already applied at previous time-step for this beam-labeling
                current_list.add(new_entry);
                // extend current beam-labeling
                // boolean breaking = false;
                for (Map.Entry<Integer, String> ment : KINSPEAK_VOCAB_IDX.entrySet()) {
                    int c = ment.getKey();
                    List<Integer> new_labeling = new ArrayList<>(labeling);
                    new_labeling.add(c);
                    // if new labeling contains duplicate char at the end, only consider paths ending with a blank
                    pr_non_blank = entry.pr_total + log_probs[c];
                    if (!labeling.isEmpty()) {
                        if (labeling.get(labeling.size() - 1) == c) {
                            pr_non_blank = entry.pr_blank + log_probs[c];
                        }
                    }
                    // fill in data
                    BeamEntry additional_entry = new BeamEntry();
                    additional_entry.labeling = new ArrayList<>(new_labeling);
                    additional_entry.pr_non_blank = logaddexp(additional_entry.pr_non_blank, pr_non_blank);
                    additional_entry.pr_total = logaddexp(additional_entry.pr_total, pr_non_blank);
                    current_list.add(additional_entry);
                }
            }
            // set new beam state
            this.last_list = current_list;
            // normalise LM scores according to beam-labeling-length
            for (BeamEntry entry : this.last_list) {
                int labeling_len = entry.labeling.size();
                entry.pr_text = (1.0 / ((labeling_len > 0) ? labeling_len : 1.0)) * entry.pr_text;
            }
            // sort by probability
            Collections.sort(this.last_list);
            return id_sequence_to_text(this.last_list.get(0).labeling);
        }
    }

}

