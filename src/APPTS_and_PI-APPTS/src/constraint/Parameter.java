package constraint;

public class Parameter {

	private int id;
	private int valueNum;

	public Parameter (int id, int valueNum){
		this.id = id;
		this.valueNum = valueNum;
	}

	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	public int getValueNum() {
		return valueNum;
	}
	public void setValueNum(int valueNum) {
		this.valueNum = valueNum;
	}

}
