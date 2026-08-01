package struct;

public class TabuElement {
	int value;
	int tenure;

	public TabuElement(int value, int tenure){
		this.value  = value;
		this.tenure = tenure;
	}

	public int getValue() {
		return value;
	}

	public void setValue(int value) {
		this.value = value;
	}

	public int getTenure() {
		return tenure;
	}

	public void setTenure(int tenure) {
		this.tenure = tenure;
	}

}
