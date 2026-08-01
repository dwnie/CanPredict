
package ftchecker;

import java.util.*;

public class FTGroup
{
    private int mDomains[];

    public ArrayList<Tuple> mAllTuples;
    public HashMap<Integer,ArrayList<ArrayList<Tuple>>> paramGroup;

    private int mNumofMFTbyParam[];

    public FTGroup(int domains[], ArrayList<Tuple> tuples)
    {

        mAllTuples = new ArrayList<Tuple>();
        paramGroup = new HashMap<Integer,ArrayList<ArrayList<Tuple>>>();
        mDomains = domains;
        mNumofMFTbyParam = new int[domains.length];
        if(tuples != null)
        {
            Tuple t;
            for(Iterator<Tuple> iterator = tuples.iterator(); iterator.hasNext(); add(t))
                t = (Tuple)iterator.next();

        }
    }

    public int size()
    {
        return mAllTuples.size();
    }

    public Tuple get(int i)
    {
        return (Tuple)mAllTuples.get(i);
    }

    public int[] getNumofMFTbyParam(){
    	return mNumofMFTbyParam;
    }

    public void calcNumofMFTbyParam(){
    	for(int param=0; param<mDomains.length; param++){
    		mNumofMFTbyParam[param] = 0;
    		ArrayList<ArrayList<Tuple>> paramList= paramGroup.get(param);
    		if(paramList!=null)
    			for(int value=0; value<mDomains[param]; value++)

    				if(paramList.get(value).size()>mNumofMFTbyParam[param])
    					mNumofMFTbyParam[param] = paramList.get(value).size();
    	}
    }

    public ArrayList<Tuple> getAllTuples()
    {
        return mAllTuples;
    }

    public void add(ArrayList<Tuple> tuples)
    {
    	for (Tuple t : tuples)
    	      add(t);

    }

    public boolean add(Tuple t)
    {
        if(contains(t) || represents(t))
            return false;
        for(Iterator<Tuple> it = mAllTuples.iterator(); it.hasNext();)
        {
            Tuple oldT = (Tuple)it.next();
            if(oldT.covers(t))
            {
                it.remove();
                for(int i = 0; i < oldT.size; i++)
                {
                    int param = oldT.getParam(i);
                    int value = oldT.getValue(i);
                    this.paramGroup.get(param).get(value).remove(oldT);
                }

            }
        }
        addWithoutCheck(t);
        return true;
    }

    public void addWithoutCheck(Tuple t)
    {
        mAllTuples.add(t);
        addToMap(t);
    }

    public void addToMap(Tuple t)
    {
        for(int i = 0; i < t.size; i++)
        {
            int param = t.getParam(i);
            int value = t.getValue(i);
            ArrayList<ArrayList<Tuple>> valueGroup = paramGroup.get(param);
            if(valueGroup == null)
            {
                valueGroup = new ArrayList<>();
                for(int j = 0; j < mDomains[param]; j++)
                    valueGroup.add(new ArrayList<>());

                paramGroup.put(param, valueGroup);
            }
            valueGroup.get(value).add(t);
        }

    }

    public boolean contains(Tuple tuple)
    {
        ArrayList<Tuple> tuples = getTuplesHasValue(tuple.getParam(0), tuple.getValue(0));
        if(tuples != null)
            return tuples.contains(tuple);
        else
            return false;
    }

    public boolean represents(Tuple tuple)
    {
        for(Iterator<Tuple> iterator = mAllTuples.iterator(); iterator.hasNext();)
        {
            Tuple t = iterator.next();
            if(tuple.covers(t))
                return true;
        }

        return false;
    }

    public boolean represents(Tuple tuple, int parameter, int value)
    {

        if(this.paramGroup.get(parameter)==null){
            return false;
        }

    	ArrayList<Tuple> tuples = getTuplesHasValue(parameter, value);
        if(tuples != null)
        	 for(Iterator<Tuple> iterator = tuples.iterator(); iterator.hasNext();)
             {
                 Tuple t = (Tuple)iterator.next();
                 if(tuple.covers(t))
                     return true;
             }

        return false;
    }

    public ArrayList<Tuple> getParaMFS(Tuple tuple, int parameter, int value)
    {
    	ArrayList<Tuple> pmft = new ArrayList<>();
    	ArrayList<Tuple> tuples = getTuplesHasValue(parameter, value);
        if(tuples != null)
        	 for(Iterator<Tuple> iterator = tuples.iterator(); iterator.hasNext();)
             {
                 Tuple t = iterator.next();
                 if(tuple.covers(t))
                     pmft.add(t);
             }

        return pmft;
    }

    public int evalMoveParaMFS(Tuple tuple, int parameter, int oldValue, int newValue)
    {
    	int num=0;
    	ArrayList<Tuple> newTuples = getTuplesHasValue(parameter, newValue);
    	ArrayList<Tuple> oldTuples = getTuplesHasValue(parameter, oldValue);
        if(oldTuples != null)
          	 for(Iterator<Tuple> iterator = oldTuples.iterator(); iterator.hasNext();)
               {
                   Tuple t = iterator.next();
                   if(tuple.covers(t))
                       num--;
               }
        tuple.setValue(parameter, newValue);
        if(newTuples != null)
        	 for(Iterator<Tuple> iterator = newTuples.iterator(); iterator.hasNext();)
             {
                 Tuple t = iterator.next();
                 if(tuple.covers(t))
                     num++;
             }
        return num;
    }

    public int countParaMFS(Tuple tuple, int parameter, int value)
    {
    	int num=0;
    	ArrayList<Tuple> tuples = getTuplesHasValue(parameter, value);
        if(tuples != null)
          	 for(Iterator<Tuple> iterator = tuples.iterator(); iterator.hasNext();)
               {
                   Tuple t = iterator.next();
                   if(tuple.covers(t))
                       num++;
               }
        return num;
    }

    public ArrayList<Tuple> getTuplesHasValue(int param, int value)
    {
        if(paramGroup.containsKey(Integer.valueOf(param)))
            return paramGroup.get(Integer.valueOf(param)).get(value);
        else
            return new ArrayList<>();
    }

    private boolean canDerive(ArrayList<ArrayList<Tuple>> valueGroup, int excludedValue)
    {
        if(valueGroup == null)
            return false;
        for(int i = 0; i < valueGroup.size(); i++)
            if(excludedValue != i && valueGroup.get(i).isEmpty())
                return false;

        return true;
    }

    public ArrayList<Tuple> derive(Tuple t)
    {
        ArrayList<Tuple> derivedTuple = new ArrayList<Tuple>();
        for(int i = 0; i < t.size; i++)
        {
            int param = t.getParam(i);
            int value = t.getValue(i);
            ArrayList<ArrayList<Tuple>> valueGroup = paramGroup.get(param);
            if(valueGroup != null && canDerive(valueGroup, value))
            {
                ArrayList<Tuple> tuples = FTUtils.getAllCombinations(param, valueGroup, value, t);
                for(Iterator<Tuple> iterator = tuples.iterator(); iterator.hasNext();)
                {
                    Tuple tuple = iterator.next();
                    if(!represents(tuple))
                        derivedTuple.add(tuple);
                }

            }
        }

        return derivedTuple;
    }

    public String toString()
    {
        return mAllTuples.toString();
    }

}
