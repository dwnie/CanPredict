package combination;

import java.util.ArrayList;

import engine.Main;

public class LineTupleSet {
	private class Entry{
		public int combinationRow;
		public int combinationColumn;
		public int column_index;
		public Entry(int combinationRow, int combinationColumn, int column_index){
			this.combinationRow = combinationRow;
			this.combinationColumn = combinationColumn;
			this.column_index = column_index;
		}
	}

	private int t_way;
	private ArrayList<Integer> lineOneCoveredCount;
	private ArrayList<ArrayList<ArrayList<ArrayList<Entry>>>> lineTupleList;
	private int [][][] mapping;

	public LineTupleSet (int caRowNum, int rowSize, int t_way){
		this.t_way = t_way;
		lineOneCoveredCount = new ArrayList<Integer>();
		lineTupleList = new ArrayList<ArrayList<ArrayList<ArrayList<Entry>>>>();
		for(int i=0; i<caRowNum; i++){
			lineOneCoveredCount.add(0);
			lineTupleList.add(new ArrayList<ArrayList<ArrayList<Entry>>>());
		}
		for(int i=0; i<caRowNum; i++){
			ArrayList<ArrayList<ArrayList<Entry>>> tmp = lineTupleList.get(i);
			for(int j=0; j<Main.paraNum; j++)
				tmp.add(new ArrayList<ArrayList<Entry>>());
		}
		for(int i=0; i<caRowNum; i++){
			for(int j=0; j<Main.paraNum; j++){
				ArrayList<ArrayList<Entry>> tmp =  lineTupleList.get(i).get(j);
				for(int k=0; k<Main.pValue[j]; k++)
					tmp.add(new ArrayList<Entry>());
			}
		}
		int [] tuple = new int[t_way];
		mapping = new int[rowSize][][];
		for(int i=0; i<rowSize; i++){
			Main.paraCombination.gett_tuple(i,tuple);
			int columnSize = 1;
			for(int j=0; j <t_way; j++)
				columnSize *= Main.pValue[tuple[j]];
			mapping[i] = new int[columnSize][];
		}
		for(int i=0; i<mapping.length; i++)
			for(int j=0; j<mapping[i].length; j++)
				mapping[i][j] = new int[t_way];
	}

	public void clear(){
		for(int i=0; i<lineOneCoveredCount.size(); i++)
			lineOneCoveredCount.set(i, 0);
		for(int i=0; i<lineTupleList.size(); i++)
			for(int j=0; j< lineTupleList.get(i).size(); j++)
				for(int k=0; k<lineTupleList.get(i).get(j).size(); k++)
					lineTupleList.get(i).get(j).get(k).clear();
		for(int i=0; i<mapping.length; i++)
			for(int j=0; j<mapping[i].length; j++)
				for(int k=0; k<mapping[i][j].length; k++)
					mapping[i][j][k] = -1;
	}

	public void push(int combinationRow, int combinationColumn, int lineIndex){
		int [] tuple = new int[t_way];
		int [] value_tuple = new int[t_way];
		Main.paraCombination.gett_tuple(combinationRow,tuple);
		Main.paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);
		int count = lineOneCoveredCount.get(lineIndex);
		lineOneCoveredCount.set(lineIndex, count+1);
		for(int i=0; i<t_way; i++){
			lineTupleList.get(lineIndex).get(tuple[i]).get(value_tuple[i])
			.add(new Entry(combinationRow,combinationColumn,i));
			if(mapping[combinationRow][combinationColumn][i] != -1){
				System.out.println("An error occured in linetuplelist!Class:LineTupleSet, method:push");
				System.exit(0);
			}
			mapping[combinationRow][combinationColumn][i] =
					lineTupleList.get(lineIndex).get(tuple[i]).get(value_tuple[i]).size()-1;
		}

	}

	public void pop(int combinationRow, int combinationColumn, int lineIndex){
		int [] tuple = new int[t_way];
		int [] value_tuple = new int[t_way];
		Main.paraCombination.gett_tuple(combinationRow,tuple);
		Main.paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);
		int count = lineOneCoveredCount.get(lineIndex);
		lineOneCoveredCount.set(lineIndex, count-1);
		for(int i=0; i<t_way; i++){
			ArrayList<Entry> list = lineTupleList.get(lineIndex).get(tuple[i]).get(value_tuple[i]);
			Entry elem = list.get(list.size()-1);
			int pos = mapping[combinationRow][combinationColumn][i];
			if(pos == -1){
				System.out.println("An error occured in linetuplelist!Class:LineTupleSet, method:pop");
				System.exit(0);
			}
			list.set(pos, elem);
			mapping[elem.combinationRow][elem.combinationColumn][elem.column_index]=pos;
			mapping[combinationRow][combinationColumn][i]=-1;
			list.remove(list.size()-1);
		}
	}

	public ArrayList<Entry> getOnecoveredListByLineVar(int lineIndex, int para, int value){
		return lineTupleList.get(lineIndex).get(para).get(value);
	}

	public int getOnecoveredCountByLineVar(int lineIndex, int para, int value){
		return lineTupleList.get(lineIndex).get(para).get(value).size();
	}

	public int getOnecoveredCount(){
		int count = 0;
		for(int i=0; i<lineOneCoveredCount.size(); i++)
			count += lineOneCoveredCount.get(i);
		return count;
	}

	public void delet_row(int lineIndex){
		ArrayList<ArrayList<ArrayList<Entry>>> elem= lineTupleList.get(lineTupleList.size()-1);
		lineTupleList.set(lineIndex, elem);
		lineTupleList.remove(lineTupleList.size()-1);

		int count = lineOneCoveredCount.get(lineOneCoveredCount.size()-1);
		lineOneCoveredCount.set(lineIndex, count);
		lineOneCoveredCount.remove(lineOneCoveredCount.size()-1);
	}

}
