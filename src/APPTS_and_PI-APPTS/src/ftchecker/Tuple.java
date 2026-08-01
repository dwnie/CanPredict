
package ftchecker;

import java.util.*;

public class Tuple
{
	private int array[];
	public int size;
    public boolean empty;

    public  Tuple(int array[])
    {
        empty = false;
        this.array = (int[])array.clone();
        size = array.length / 2;
        if(size == 0)
            empty = true;
    }

    public Tuple(List<Integer> list)
    {
        empty = false;
        size = list.size() / 2;
        if(list.size() == 0)
        {
            empty = true;
        } else
        {
            array = new int[list.size()];
            for(int i = 0; i < list.size(); i++)
                array[i] = ((Integer)list.get(i)).intValue();

        }
    }

    public int getParam(int i)
    {
        return array[2 * i];
    }

    public int getValue(int i)
    {
        return array[2 * i + 1];
    }

    public void setValue(int i, int value)
    {
    	array[2 * i + 1] = value;
    }

    public int getParamValue(int p)
    {
        for(int i = 0; i < size; i++)
            if(getParam(i) == p)
                return getValue(i);

        return -1;
    }

    public boolean covers(Tuple t2)
    {
        if(t2 == null)
            return true;
        if(t2.size > size)
            return false;
        Tuple t1 = this;
        int i = 0;
        for(int j = 0; j < t2.size; j++)
        {
            while(i < size && t1.getParam(i) < t2.getParam(j))
                i++;
            if(i >= size || t1.getParam(i) != t2.getParam(j) || t1.getValue(i) != t2.getValue(j))
                return false;
            i++;
        }

        return true;
    }

    public boolean isEqual(Tuple t2)
    {
    	if(t2.size != size)
    		return false;
    	for(int i=0; i<t2.size; i++)
    		if(this.getParam(i)!=t2.getParam(i) || this.getValue(i) != t2.getValue(i))
    			return false;
    	return true;
    }

    public Tuple combine(Tuple t2, int param)
    {
        if(t2 == null || t2.size == 0)
            return null;
        Tuple t1 = this;
        int i = 0;
        int j = 0;
        int k = 0;
        int newArray[] = new int[2 * t1.size + 2 * t2.size];
        while(i < t1.size || j < t2.size)
            if(i < t1.size && t1.getParam(i) == param)
                i++;
            else if(j < t2.size && t2.getParam(j) == param)
                j++;
            else
            {
                if(i >= t1.size)
                {
                    newArray[2 * k] = t2.getParam(j);
                    newArray[2 * k + 1] = t2.getValue(j);
                    j++;
                }
                else if(j >= t2.size)
                {
                    newArray[2 * k] = t1.getParam(i);
                    newArray[2 * k + 1] = t1.getValue(i);
                    i++;
                }
                else if(t1.getParam(i) < t2.getParam(j))
                {
                    newArray[2 * k] = t1.getParam(i);
                    newArray[2 * k + 1] = t1.getValue(i);
                    i++;
                }
                else if(t1.getParam(i) > t2.getParam(j))
                {
                    newArray[2 * k] = t2.getParam(j);
                    newArray[2 * k + 1] = t2.getValue(j);
                    j++;
                }
                else
                {
                    if(t1.getValue(i) != t2.getValue(j))
                        return null;
                    newArray[2 * k] = t1.getParam(i);
                    newArray[2 * k + 1] = t1.getValue(i);
                    i++;
                    j++;
                }
                k++;
            }
        return new Tuple(Arrays.copyOf(newArray, 2 * k));
    }

    public String toString()
    {
        StringBuffer s = new StringBuffer("(");
        for(int i = 0; i < size; i++)
        {
            s.append(getParam(i));
            s.append(".");
            s.append(getValue(i));
            if(i != size - 1)
                s.append("|");
        }

        s.append(")");
        return s.toString();
    }
}
