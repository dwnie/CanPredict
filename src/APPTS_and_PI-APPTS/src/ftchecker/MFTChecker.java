package ftchecker;

import java.util.*;

// Derives the closure of minimum forbidden tuples and uses it for incremental
// validity and move-effect queries.
public class MFTChecker implements IFTChecker {

    private int[] mDomains;
    private FTGroup mAllMFTs;
    private ArrayList<Tuple> mInputTuples;

    public MFTChecker() {
    }

    public void init(int[] domains) {
        FTUtils.nParams = domains.length;
        mDomains = domains;
        mInputTuples = new ArrayList<Tuple>();
    }

    public void addForbiddenTuple(int[] tuple) {
        mInputTuples.add(new Tuple(tuple));
    }

    public void addForbiddenTuples(ArrayList<Tuple> tuples) {
        mInputTuples.addAll(tuples);
    }

    public List<Tuple> getMinimumForbiddenTuples() {
        if (mAllMFTs == null)
            mAllMFTs = deriveAll(mInputTuples);
        return mAllMFTs.getAllTuples();
    }

    public void genMinimumForbiddenTuples() {
        if (mAllMFTs == null)
            mAllMFTs = deriveAll(mInputTuples);
    }

    public int[] getNumofMFTbyParam() {
        if (mAllMFTs == null)
            mAllMFTs = deriveAll(mInputTuples);
        return mAllMFTs.getNumofMFTbyParam();
    }

    private FTGroup deriveAll(ArrayList<Tuple> inputTuples) {
        long start = System.currentTimeMillis();
        FTGroup group = new FTGroup(mDomains, null);
        if (inputTuples.isEmpty()) {
            System.out.println("No input forbidden tuples.");
            return group;
        }
        PriorityQueue<Tuple> queue = new PriorityQueue<Tuple>(inputTuples.size(), new Comparator<Tuple>() {

            public int compare(Tuple t1, Tuple t2) {
                if (t1.size != t2.size)
                    return t1.size - t2.size;
                for (int i = 0; i < t1.size; i++) {
                    int p1 = t1.getParam(i);
                    int v1 = t1.getValue(i);
                    int p2 = t2.getParam(i);
                    int v2 = t2.getValue(i);
                    if (p1 != p2)
                        return p1 - p2;
                    if (v1 != v2)
                        return v1 - v2;
                }

                return 0;
            }
        });
        queue.addAll(inputTuples);
        // Smaller tuples are processed first so redundant supersets can be rejected
        // while newly derived tuples are fed back into the same work queue.
        while (!queue.isEmpty()) {
            Tuple t = queue.poll();
            if (group.add(t)) {
                ArrayList<Tuple> derivedTuples = group.derive(t);
                queue.addAll(derivedTuples);
            }
        }
        group.calcNumofMFTbyParam();
        long time = System.currentTimeMillis() - start;

        return group;
    }

    public ArrayList<Tuple> getMFTbyParam(int parameter, int value){
        if (mAllMFTs == null)
            mAllMFTs = deriveAll(mInputTuples);
        return mAllMFTs.getTuplesHasValue(parameter,value);
    }

    public boolean isValid(int[] test) {
        if (mAllMFTs == null)
            mAllMFTs = deriveAll(mInputTuples);
        Tuple tuple = new Tuple(test);
        return !mAllMFTs.represents(tuple);
    }

    public boolean isValid(int[] test, int parameter, int value) {
        if (mAllMFTs == null)
            mAllMFTs = deriveAll(mInputTuples);
        Tuple tuple = new Tuple(test);
        return !mAllMFTs.represents(tuple, parameter, value);
    }

    public ArrayList<Tuple> getMFTbyParam(int[] test, int parameter, int value) {
        if (mAllMFTs == null)
            mAllMFTs = deriveAll(mInputTuples);
        Tuple tuple = new Tuple(test);
        return mAllMFTs.getParaMFS(tuple, parameter, value);
    }

    public int getMFTNumbyParam(int[] test, int parameter, int value) {
        int num;
        if (mAllMFTs == null)
            mAllMFTs = deriveAll(mInputTuples);
        Tuple tuple = new Tuple(test);
        num = mAllMFTs.countParaMFS(tuple, parameter, value);
        return num;
    }

    public int evalMove_ParaMFS(int[] test, int parameter, int oldValue, int newValue) {
        int num1, num2;
        int[] newtest = new int[test.length];
        System.arraycopy(test, 0, newtest, 0, test.length);
        newtest[2 * parameter + 1] = newValue;
        if (mAllMFTs == null)
            mAllMFTs = deriveAll(mInputTuples);
        Tuple tuple = new Tuple(test);
        Tuple newtuple = new Tuple(newtest);
        num1 = mAllMFTs.countParaMFS(tuple, parameter, oldValue);
        num2 = mAllMFTs.countParaMFS(newtuple, parameter, newValue);

        return num2 - num1;
    }

    public boolean isValid(ArrayList<Integer> test, int parameter, int value) {
        if (mAllMFTs == null)
            mAllMFTs = deriveAll(mInputTuples);
        Tuple tuple = new Tuple(test);
        return !mAllMFTs.represents(tuple, parameter, value);
    }

}
