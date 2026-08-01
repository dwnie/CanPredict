package tabulist;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import engine.Main;
import struct.Element;

public class TabuList {

    private final Element[] modifiedList;

    private final int MaxLen;

    private int front;

    private int rear;

    private final ArrayList<Element> tablst;

    public TabuList(int tabuLen) {

        MaxLen = tabuLen + 1;
        modifiedList = new Element[MaxLen];
        for (int i = 0; i < MaxLen; i++)
            modifiedList[i] = new Element();
        front = rear = 0;
        tablst = new ArrayList<>();
    }

    public Element getElement(int p) {
        int k;
        k = (front + p) % MaxLen;
        return modifiedList[k];
    }

    public void clear() {
        front = rear = 0;
        tablst.clear();
    }

    public void updateUndoList(int row, int column, int value) {
        if ((rear + 1) % MaxLen == front) {
            front = (front + 1) % MaxLen;
        }
        modifiedList[rear].row = row;
        modifiedList[rear].column = column;
        modifiedList[rear].value = value;
        rear = (rear + 1) % MaxLen;
    }

    public void getTabuList() {
        int i, row, column, value, length;

        length = (rear - front + MaxLen) % MaxLen;
        if (length == 0)
            return;

        HashMap<Integer, Integer> changes = new HashMap<Integer, Integer>();

        tablst.clear();

        int stateCount = 0;
        for (i = length - 1; i >= 0; i--) {

            row = getElement(i).row;
            column = getElement(i).column;
            value = getElement(i).value;

            if (Main.coverageArray[row][column] == value) {
                changes.remove(row * Main.paraNum + column);
                stateCount -= 1;
            } else {
                if (!changes.containsKey(row * Main.paraNum + column))
                    stateCount += 1;
                changes.put(row * Main.paraNum + column, value);
            }

            if (stateCount == 1)
                addTabuElement(changes);
        }

    }

    private void addTabuElement(HashMap<Integer, Integer> changes) {
        Iterator<Entry<Integer, Integer>> iter = changes.entrySet().iterator();
        Entry<Integer, Integer> entry = iter.next();
        Integer key = entry.getKey();
        Integer value = entry.getValue();
        int row = key / Main.paraNum;
        int column = key % Main.paraNum;
        tablst.add(new Element(row, column, value));
    }

    public boolean isTabu(int modiRow, int modiColumn, int modiValue) {
        for (int i = 0; i < tablst.size(); i++) {
            Element e = tablst.get(i);
            if (e.row == modiRow && e.column == modiColumn && e.value == modiValue)
                return true;
        }
        return false;
    }

}
