package combination;

import java.util.ArrayList;

import struct.CombinationPosition;

public class TupleSet {
    private final int t_way;
    private final ArrayList<CombinationPosition> tupleList;

    public TupleSet(int t_way) {
        this.t_way = t_way;
        tupleList = new ArrayList<>();

    }

    public void clear() {
        tupleList.clear();
    }

    public synchronized void push(int combinationRow, int combinationColumn) {
        tupleList.add(new CombinationPosition(combinationRow, combinationColumn));
    }

    public synchronized void pop(int combinationRow, int combinationColumn) {

        int pos = 0;
        while (pos < tupleList.size()) {
            CombinationPosition entry = tupleList.get(pos);
            if (entry.equals(new CombinationPosition(combinationRow, combinationColumn)))
                break;
            pos++;
        }
        if (pos >= tupleList.size()) {
            System.out.println("It cannot find the specified entry");
            System.exit(0);
        }

        if (pos != tupleList.size() - 1) {
            CombinationPosition lastElement = tupleList.get(tupleList.size() - 1);
            tupleList.set(pos, lastElement);
        }
        tupleList.remove(tupleList.size() - 1);

    }

    public ArrayList<CombinationPosition> getTupleList() {
        return tupleList;
    }

    public int getSize() {
        return tupleList.size();
    }

}
