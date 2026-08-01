package combination;

import java.util.ArrayList;

import engine.Main;
import struct.CombinationPosition;

public class VarTupleSet {

    private final Object lock = new Object();
    private final int t_way;
    private final ArrayList<ArrayList<ArrayList<CombinationPosition>>> varTupleList;

    private int onceCoveredCount;

    public VarTupleSet(int t_way) {
        onceCoveredCount = 0;
        this.t_way = t_way;
        varTupleList = new ArrayList<>();
        for (int i = 0; i < Main.paraNum; i++) {
            varTupleList.add(new ArrayList<>());
        }
        for (int i = 0; i < Main.paraNum; i++) {
            ArrayList<ArrayList<CombinationPosition>> tmp = varTupleList.get(i);
            for (int j = 0; j < Main.pValue[i]; j++)
                tmp.add(new ArrayList<>());
        }

    }

    public void clear() {
        onceCoveredCount = 0;
        for (int i = 0; i < varTupleList.size(); i++)
            for (int j = 0; j < varTupleList.get(i).size(); j++)
                varTupleList.get(i).get(j).clear();
    }

    public void push(int combinationRow, int combinationColumn) {
        int[] tuple = new int[t_way];
        int[] value_tuple = new int[t_way];
        Main.paraCombination.gett_tuple(combinationRow, tuple);
        Main.paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);

        synchronized (lock) {
            for (int i = 0; i < t_way; i++) {
                varTupleList.get(tuple[i]).get(value_tuple[i]).add(new CombinationPosition(combinationRow, combinationColumn));
            }
            onceCoveredCount++;
        }

    }

    public void pop(int combinationRow, int combinationColumn) {
        int[] tuple = new int[t_way];
        int[] value_tuple = new int[t_way];
        Main.paraCombination.gett_tuple(combinationRow, tuple);
        Main.paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);

        synchronized (lock) {
            for (int i = 0; i < t_way; i++) {
                ArrayList<CombinationPosition> list = varTupleList.get(tuple[i]).get(value_tuple[i]);

                int pos = 0;
                while (pos < list.size()) {
                    CombinationPosition entry = list.get(pos);
                    if (entry.equals(new CombinationPosition(combinationRow, combinationColumn)))
                        break;
                    pos++;
                }
                if (pos >= list.size()) {
                    System.out.println("It cannot find the specified entry");
                    System.exit(0);
                }

                if (pos != list.size() - 1) {
                    CombinationPosition lastElement = list.get(list.size() - 1);
                    list.set(pos, lastElement);
                }
                list.remove(list.size() - 1);
            }
            onceCoveredCount--;
        }
    }

    public ArrayList<CombinationPosition> getOnceCoveredListByVar(int para, int value) {
        return varTupleList.get(para).get(value);
    }

    public int getOnceCoveredCountByVar(int para, int value) {
        return varTupleList.get(para).get(value).size();
    }

    public int getOnceCoveredCount() {
        return onceCoveredCount;
    }

}
