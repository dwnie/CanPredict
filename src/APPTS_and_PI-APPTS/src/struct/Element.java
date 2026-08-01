package struct;

public class Element {

	public int row;
	public int column;
	public int value;
	public double effect;
	public boolean isfeasible;

	public int effectMFT;
	public int effectUncovered;
	public boolean istabu;

	public Element(){
	}
	public Element(int row, int column, int value){
		this.row = row;
		this.column = column;
		this.value = value;
	}
	public Element(int row, int column, int value, double effect, boolean isfeasible){
		this.row = row;
		this.column = column;
		this.value = value;
		this.effect = effect;
		this.isfeasible = isfeasible;
	}

	public Element(int row, int column, int value, int effectMFT, int effectUncovered, boolean istabu){
		this.row = row;
		this.column = column;
		this.value = value;
		this.effectMFT = effectMFT;
		this.effectUncovered = effectUncovered;
		this.istabu = istabu;
	}

}
