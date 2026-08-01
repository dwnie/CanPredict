
package ftchecker;

import java.util.ArrayList;
import java.util.List;

public interface IFTChecker
{

    void init(int ai[]);

    void addForbiddenTuple(int ai[]);

    void addForbiddenTuples(ArrayList<Tuple> tuples);

    List<Tuple> getMinimumForbiddenTuples();

    void genMinimumForbiddenTuples();

    ArrayList<Tuple> getMFTbyParam(int parameter, int value);

    boolean isValid(int ai[]);

    boolean isValid(int ai[], int parameter, int value);

    boolean isValid(ArrayList<Integer> test, int parameter, int value);

    ArrayList<Tuple> getMFTbyParam(int test[], int parameter, int value);

    int getMFTNumbyParam(int test[], int parameter, int value);

    int evalMove_ParaMFS(int[] test, int parameter, int oldValue, int newValue);

    int[] getNumofMFTbyParam();
}
