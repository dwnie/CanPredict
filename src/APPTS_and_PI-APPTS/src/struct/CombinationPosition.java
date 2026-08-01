package struct;

public class CombinationPosition  implements Comparable<CombinationPosition>{

	public int combinationRow;

	public int combinationColumn;

	public CombinationPosition(int combinationRow, int combinationColumn){
		this.combinationRow = combinationRow;
		this.combinationColumn = combinationColumn;
	}

	@Override
	public boolean equals(Object object){
		boolean result=false;
		if(object==this)
			return true;
		else{
				if(object!=null && object instanceof CombinationPosition){
					CombinationPosition position = (CombinationPosition) object;
					return this.combinationRow==position.combinationRow &&
							this.combinationColumn==position.combinationColumn;
			}
			return result;
		}

	}

	@Override
	public int compareTo(CombinationPosition o) {

		return 0;
	}

}
