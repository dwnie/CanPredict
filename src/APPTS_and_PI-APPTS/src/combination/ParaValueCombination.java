package combination;

import engine.Main;

public class ParaValueCombination {
	private	int t_way;

	public	ParaValueCombination(int t_way){
		this.t_way = t_way;
	}
	public	int getColumnNum(int[] paraTuple,int []tuple){
		int i,count=0;
		for(i=0; i<t_way; i++)
			count = count*Main.pValue[paraTuple[i]]+tuple[i];
		return count;

	}
	public	void gett_tuple(int[] paraTuple,int column, int []tuple){
		for(int i=t_way-1; i>=0; i--)
		{
			tuple[i] = column%Main.pValue[paraTuple[i]];
			column = column/Main.pValue[paraTuple[i]];
		}
	}

}
