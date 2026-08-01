package engine;

import java.io.*;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.Random;

import parallel.*;
import struct.CombinationPosition;
import struct.Element;
import tabulist.TabuList;
import combination.CoveredCombination;
import combination.ParaCombination;
import combination.ParaValueCombination;
import constraint.ForbiddenTuples;
import evalue.EvalMove;
import ftchecker.IFTChecker;
import ftchecker.MFTChecker;
import ftchecker.MFTinCA;
import ftchecker.Tuple;

public class Main {

    public static int tabuLen = 200;

    public static int[][] coverageArray;

    public static int t_way;

    public static int paraNum;

    public static int[] pValue;

    public static int rowNum;

    public static ParaCombination paraCombination;

    public static ParaValueCombination paraValueCombination;

    public static CoveredCombination coveredCombination;

    public static IFTChecker checker;

    public static int[] numMFTParam;

    public static MFTinCA mft_ca;

    public static TabuList tabuList = null;

    public static EvalMove evalMove = null;

    public static double lamd;
    public static double beta1, beta2;
    public static double MINLAMD = 0.01;
    public static double MAXLAMD = 50.0;
    public static double INILAMD = 4.0;
    public static double BETA1 = 1.5;
    public static double BETA2 = 1.2;

    public static void main(String[] args) throws IOException {
        int i, j;
        int maxRowNum;
        int minRowNum;
        int iter;
        int uncoveredCount;
        int mft_ca_count;
        double fitness;
        double minFitness;
        int[][] bestArray;
        long finish;

        int cutoffTime;
        String modelFileName;
        String constraintsFileName;

        int iterNum = 5000;

        if (args.length < 4) {
            System.out.println("Please provide cutoffTime, model, constraint file, coverage strength");
            System.exit(0);
        }

        cutoffTime = Integer.valueOf(args[0]);

        modelFileName = "./model/" + args[1];
        constraintsFileName = "./model/" + args[2];

        t_way = Integer.valueOf(args[3]);

        if (args.length > 4)
          iterNum = Integer.valueOf(args[4]);

        // The quotient selects serial or parallel move evaluation; the remainder
        // selects serial or parallel coverage updates.
        int parallelMode = 2;
        int evalMode = parallelMode / 2;

        int updateMode = parallelMode % 2;

        checker = new MFTChecker();

        pValue = ForbiddenTuples.getDomains(modelFileName);

        ArrayList<Integer> indexToParamMap = ForbiddenTuples.getIndexToParamMap(pValue);
        ArrayList<Integer> indexToValueMap = ForbiddenTuples.getIndexToValueMap(pValue);

        ArrayList<Tuple> forbiddenTuples = ForbiddenTuples.getForbiddenTuples(constraintsFileName, indexToParamMap,
                indexToValueMap);

        checker.init(pValue);

        checker.addForbiddenTuples(forbiddenTuples);

        // Expand the input constraints into the minimum forbidden tuples used by
        // validity checks and move evaluation.
        checker.genMinimumForbiddenTuples();

        numMFTParam = checker.getNumofMFTbyParam();

        int max_numMFTParam = numMFTParam[0];
        for (i = 1; i < paraNum; i++)
            if (max_numMFTParam < numMFTParam[i])
                max_numMFTParam = numMFTParam[i];

        int rowSize;
        int num = 1;
        int denum = 1;
        int para = paraNum;
        int way = t_way;
        for (i = 0; i < t_way; i++) {
            num *= (para--);
            denum *= (way--);
        }
        rowSize = num / denum;

        paraCombination = new ParaCombination(rowSize, t_way);
        paraValueCombination = new ParaValueCombination(t_way);
        coveredCombination = new CoveredCombination(rowSize, pValue, t_way);
        evalMove = new EvalMove(t_way);
        tabuList = new TabuList(tabuLen);

        System.out.println("Wait......");

        double effect;
        double bestEffect;
        ArrayList<Element> moveSet;
        ArrayList<Element> bestMoveSet;
        moveSet = new ArrayList<>();
        bestMoveSet = new ArrayList<>();

        boolean isSuccess = false;
        int[] tuple = new int[t_way];
        int[] value_tuple = new int[t_way];
        Element changedElement = new Element();

        long wholeStart = System.currentTimeMillis();

        // IPOG supplies the initial valid covering array for the tabu-search phase.
        maxRowNum = initCoverageArrayByIpog(modelFileName, constraintsFileName, forbiddenTuples.size(), t_way);
        bestArray = new int[maxRowNum][paraNum];
        rowNum = maxRowNum;
        minRowNum = maxRowNum;
        double  wholeDuration = (double) (System.currentTimeMillis() - wholeStart) / 1000;
        System.out.println("Success to construct an initial CA with " + minRowNum + " test cases in " +
                wholeDuration
                + " seconds");
        for (i = 0; i < rowNum; i++)
            for (j = 0; j < paraNum; j++)
                bestArray[i][j] = coverageArray[i][j];

        coveredCombination.init(rowNum);

        // Standard APPTS removes one row. PI-APPTS uses startSize to remove several
        // rows before the first repair attempt.
        int firstDeleteRows = 1;

        if(args.length > 5 ) {
            int startSize = Integer.valueOf(args[5]);
            firstDeleteRows = rowNum - startSize > 1 ? rowNum - startSize : 1;
        }

        // Rows removed by PI-APPTS are retained so an overly optimistic prediction
        // can be repaired by restoring rows one at a time.
        Deque<int[]> stack = new ArrayDeque<>();

        boolean isFirstRound = true;

        while ((System.currentTimeMillis() - wholeStart) / 1000 < cutoffTime) {

            Random rand = new Random();

            isSuccess = false;

            int deleteRows = 1;
            if(isFirstRound){
                deleteRows  =  firstDeleteRows;
            }

            // The first PI-APPTS reduction is deterministic; later APPTS iterations
            // remove a random row.
            for(int delIter = 1; delIter<=deleteRows; delIter++){

                int deletRowNum = rowNum-1;

                if(!isFirstRound)
                    deletRowNum = rand.nextInt(rowNum);

                coveredCombination.delete_row(deletRowNum);

                if(isFirstRound && deleteRows>1){
                    stack.push(coverageArray[deletRowNum]);
                }

                if(deletRowNum != rowNum-1)
                    for (j = 0; j < paraNum; j++)
                        coverageArray[deletRowNum][j] = coverageArray[rowNum - 1][j];
                rowNum--;
            }
            System.out.println("delete "+ deleteRows + " rows");

            uncoveredCount = coveredCombination.getUncoveredCount();
            if (uncoveredCount == 0) {
                isSuccess = true;
                minRowNum = rowNum;
                for (i = 0; i < rowNum; i++)
                    for (j = 0; j < paraNum; j++)
                        bestArray[i][j] = coverageArray[i][j];
            }

            mft_ca = new MFTinCA(rowNum, checker, numMFTParam);
            mft_ca_count = 0;

            tabuList.clear();

            lamd = INILAMD;
            beta1 = BETA1;
            beta2 = BETA2;

            fitness = uncoveredCount + lamd * mft_ca_count;
            minFitness = fitness;

            boolean islastfeasible = false;
            boolean isfeasible;
            int nf = 0;
            boolean isfeasiblebegin = false;
            int nnotimpr = 0;

            iter = 0;

            while (!isSuccess && (System.currentTimeMillis() - wholeStart) / 1000 < cutoffTime)
            {

                moveSet.clear();
                bestMoveSet.clear();

                bestEffect = rowSize + lamd * max_numMFTParam;

                tabuList.getTabuList();

                mft_ca_count = mft_ca.getCount();
                uncoveredCount = coveredCombination.getUncoveredCount();

                // Repair forbidden-tuple violations first. Once the candidate is
                // feasible, repair uncovered interactions instead.
                if (mft_ca_count > 0) {

                    int[] row_mft = new int[mft_ca_count];
                    ArrayList<Tuple> tuple_mfts = mft_ca.getMFTs(row_mft);

                    int rp = rand.nextInt(mft_ca_count);

                    Tuple tuple_mft = tuple_mfts.get(rp);
                    int size = tuple_mft.size;
                    int[] tuple_param = new int[size];
                    int[] tuple_value = new int[size];
                    int neighborsCount = 0;
                    for (i = 0; i < size; i++) {
                        tuple_param[i] = tuple_mft.getParam(i);
                        tuple_value[i] = tuple_mft.getValue(i);
                        neighborsCount += pValue[tuple_param[i]] - 1;
                    }

                    Element[] neighbors = new Element[neighborsCount];
                    ArrayList<NeighborEval> neighborlist = new ArrayList<>();
                    int no = 0;
                    for (int z = 0; z < size; z++) {
                        for (int v = 0; v < pValue[tuple_param[z]]; v++) {
                            if (v == tuple_value[z])
                                continue;

                            if (evalMode == 0 || evalMode == 1)
                                neighborlist.add(
                                        new NeighborEvalSerial1(row_mft[rp], tuple_param[z], v, neighbors, no++));

                            else
                                neighborlist.add(
                                        new NeighborEvalParallel1(row_mft[rp], tuple_param[z], v, neighbors, no++));
                        }
                    }

                    if (evalMode == 0 || evalMode == 2) {
                        for (NeighborEval neighbor : neighborlist)
                            neighbor.caculEffcAndTabu();
                    }

                    else
                        neighborlist.parallelStream().forEach(NeighborEval::caculEffcAndTabu);

                    for (i = 0; i < neighborsCount; i++) {

                        Element elem = neighbors[i];
                        int effect1 = elem.effectMFT;
                        int effect2 = elem.effectUncovered;
                        effect = effect1 * lamd + effect2;

                        Element move;
                        if (effect1 + mft_ca_count == 0)
                            move = new Element(elem.row, elem.column, elem.value, effect, true);
                        else
                            move = new Element(elem.row, elem.column, elem.value, effect, false);
                        moveSet.add(move);

                        if ((mft_ca_count + effect1 > 0 || uncoveredCount + effect2 > 0) && elem.istabu)
                            continue;

                        if (bestEffect >= effect) {

                            if (bestEffect > effect) {
                                bestMoveSet.clear();
                                bestEffect = effect;
                            }
                            isfeasible = move.isfeasible;
                            Element bestMove = new Element(elem.row, elem.column, elem.value, effect, isfeasible);
                            bestMoveSet.add(bestMove);
                        }
                    }
                } else {

                    int effect1;
                    int effect2;

                    int combinationRow;
                    int combinationColumn;
                    ArrayList<CombinationPosition> uncoveredcombinations = coveredCombination.getUncoveredList();

                    int rp = rand.nextInt(uncoveredcombinations.size());
                    CombinationPosition cp = uncoveredcombinations.get(rp);
                    combinationColumn = cp.combinationColumn;
                    combinationRow = cp.combinationRow;
                    paraCombination.gett_tuple(combinationRow, tuple);
                    paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);

                    Element[] neighbors = new Element[rowNum];

                    int neighborsCount = 0;

                    ArrayList<NeighborEval> neighborlist = new ArrayList<>();

                    for (int z = 0; z < rowNum; z++) {

                        int difCount = 0;
                        for (int m = 0; m < t_way; m++) {
                            if (coverageArray[z][tuple[m]] != value_tuple[m]) {
                                if (difCount == 0) {
                                    changedElement.row = z;
                                    changedElement.column = tuple[m];
                                    changedElement.value = value_tuple[m];
                                }
                                difCount++;
                            }
                            if (difCount >= 2)
                                break;
                        }
                        if (difCount >= 2)
                            continue;

                        if (evalMode == 0 || evalMode == 1)
                            neighborlist.add(
                                    new NeighborEvalSerial2(z, changedElement.column, changedElement.value, neighbors,
                                            neighborsCount++));

                        else
                            neighborlist.add(
                                    new NeighborEvalParallel2(z, changedElement.column, changedElement.value, neighbors,
                                            neighborsCount++));
                    }

                    if (neighborsCount == 0) {

                        continue;
                    }

                    if (evalMode == 0 || evalMode == 2) {
                        for (NeighborEval neighbor : neighborlist)
                            neighbor.caculEffcAndTabu();
                    }

                    else
                        neighborlist.parallelStream().forEach(NeighborEval::caculEffcAndTabu);

                    for (int z = 0; z < neighborsCount; z++) {

                        Element elem = neighbors[z];
                        effect1 = elem.effectMFT;
                        effect2 = elem.effectUncovered;

                        effect = effect1 * lamd + effect2;

                        Element move;
                        if (effect1 + mft_ca_count == 0)
                            move = new Element(elem.row, elem.column, elem.value, effect, true);
                        else
                            move = new Element(elem.row, elem.column, elem.value, effect, false);
                        moveSet.add(move);

                        if ((mft_ca_count + effect1 > 0 || uncoveredCount + effect2 > 0) && elem.istabu)
                            continue;

                        if (bestEffect >= effect) {

                            if (bestEffect > effect) {
                                bestMoveSet.clear();
                                bestEffect = effect;
                            }
                            isfeasible = move.isfeasible;
                            Element bestMove = new Element(elem.row, elem.column, elem.value, effect, isfeasible);
                            bestMoveSet.add(bestMove);
                        }
                    }
                }

                int movep;
                Element moveElement;

                if (bestMoveSet.size() > 0) {
                    movep = rand.nextInt(bestMoveSet.size());
                    moveElement = bestMoveSet.get(movep);

                } else {
                    movep = rand.nextInt(moveSet.size());
                    moveElement = moveSet.get(movep);

                }

                // Adapt the penalty for forbidden-tuple violations according to
                // consecutive feasible or infeasible moves.
                if (iter == 0) {
                    nf = 1;
                    islastfeasible = moveElement.isfeasible;

                } else {
                    if (moveElement.isfeasible) {
                        if (islastfeasible) {
                            nf++;
                            if (nf >= 10 * rowNum) {
                                lamd = lamd / beta2;
                                if (lamd < MINLAMD)
                                    lamd = MINLAMD;
                                nf = 0;
                            }
                        } else {
                            nf = 1;
                            islastfeasible = true;
                        }
                    } else {
                        if (!islastfeasible) {
                            nf++;
                            if (nf >= 10 * rowNum) {
                                lamd = lamd * beta1;
                                if (lamd > MAXLAMD)
                                    lamd = MAXLAMD;
                                nf = 0;
                            }
                        } else {
                            nf = 1;
                            islastfeasible = false;
                        }
                    }

                }

                tabuList.updateUndoList(moveElement.row, moveElement.column,
                        coverageArray[moveElement.row][moveElement.column]);

                mft_ca.deletMFT(moveElement.column, moveElement.row);
                mft_ca.addMFT(moveElement.value, moveElement.column, moveElement.row);

                int[][] tuple_array = evalMove.getParaCombsByPara(moveElement.column);

                ArrayList<CombinationUpdate> combinationList = new ArrayList<>();
                for (int index = 0; index < tuple_array.length; index++) {

                    combinationList.add(new CombinationUpdate(moveElement.row, tuple_array[index]));

                    combinationList.add(new CombinationUpdate(moveElement.row, moveElement.column, moveElement.value,
                            tuple_array[index]));
                }

                if (updateMode == 0) {
                    for (CombinationUpdate combinationUpdate : combinationList)
                        combinationUpdate.update();
                }

                else {
                    combinationList.parallelStream().forEach(CombinationUpdate::update);
                }

                coverageArray[moveElement.row][moveElement.column] = moveElement.value;

                mft_ca_count = mft_ca.getCount();
                uncoveredCount = coveredCombination.getUncoveredCount();

                if (mft_ca_count == 0 && uncoveredCount == 0) {
                    for (i = 0; i < rowNum; i++)
                        for (j = 0; j < paraNum; j++)
                            bestArray[i][j] = coverageArray[i][j];

                    isSuccess = true;
                } else {

                    if (iter == 0) {
                        minFitness = uncoveredCount;
                        isfeasiblebegin = true;
                        nnotimpr = 0;
                    } else {
                        if (minFitness > uncoveredCount) {
                            minFitness = uncoveredCount;
                            nnotimpr = 0;
                        } else
                            nnotimpr++;
                    }

                }

                // On stagnation, PI-APPTS backtracks with a stored row; standard
                // APPTS diversifies by replacing one row with a generated candidate.
                if (nnotimpr > iterNum && uncoveredCount > 0) {
                    nnotimpr = 0;

                    tabuList.clear();

                    if(isFirstRound && !stack.isEmpty()){
                        System.out.println("add one row");
                        int[] newrow = stack.pop();
                        mft_ca.addNoMFT_row();
                        coveredCombination.add_row(newrow);
                        for (j = 0; j < paraNum; j++)
                            coverageArray[rowNum][j] = newrow[j];
                        rowNum++;
                    }

                    else {

                        int[] newrow = generateOneRowCover(rand.nextInt(uncoveredCount));
                        int changeRowNo = rand.nextInt(rowNum);
                        mft_ca.updateMFT_row(changeRowNo, newrow);
                        coveredCombination.update_row(changeRowNo, newrow);
                        for (j = 0; j < paraNum; j++)
                            coverageArray[changeRowNo][j] = newrow[j];
                    }

                    mft_ca_count = mft_ca.getCount();
                    uncoveredCount = coveredCombination.getUncoveredCount();

                    if (mft_ca_count == 0 && uncoveredCount == 0) {
                        for (i = 0; i < rowNum; i++)
                            for (j = 0; j < paraNum; j++)
                                bestArray[i][j] = coverageArray[i][j];

                        isSuccess = true;
                    }

                }

                iter++;
            }

            isFirstRound = false;

            if (!isSuccess)
                break;

            minRowNum = rowNum;
            finish = System.currentTimeMillis();
            wholeDuration = (double) (finish - wholeStart) / 1000;

        }

        System.out.println(
                                        "Success to construct a CA with " +  minRowNum + " test cases in " + wholeDuration + " seconds");

        BufferedWriter bw = new BufferedWriter(new FileWriter("result.txt"));
        bw.write("Construct a CA with " +  minRowNum + " test cases in " + wholeDuration + " seconds");
        bw.newLine();
        for (i = 0; i < minRowNum; i++) {
            for (j = 0; j < paraNum; j++)
                bw.write(bestArray[i][j] + " ");
            bw.newLine();
        }
        bw.flush();
        bw.close();

        System.out.println("End");

    }

    public static int[] generateOneRowCover(int p) {
        int i, j, param;
        int[] oneRow = new int[paraNum];
        int[] tuple = new int[t_way];
        int[] value_tuple = new int[t_way];
        int combinationRow;
        int combinationColumn;
        CombinationPosition uncoveredpaire;
        uncoveredpaire = coveredCombination.getUncoveredCombByList(p);
        combinationRow = uncoveredpaire.combinationRow;
        combinationColumn = uncoveredpaire.combinationColumn;
        paraCombination.gett_tuple(combinationRow, tuple);
        paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);

        Random rand = new Random();
        for (j = 0; j < t_way; j++) {
            oneRow[tuple[j]] = value_tuple[j];
        }
        for (param = 0, j = t_way; param < paraNum; param++) {
            for (i = 0; i < t_way; i++)
                if (param == tuple[i])
                    break;
            if (i < t_way)
                continue;

            int value = rand.nextInt(pValue[param]);
            oneRow[param] = value;
        }
        return oneRow;
    }

    public static int initCoverageArrayByIpog(String modelFileName, String constraintsFileName, int ftcount,
            int t_way) throws IOException {
        String t_way_string = String.valueOf(t_way);

        // IPOG expects the interaction strength as the first model-file line, so a
        // temporary model is created without modifying the original input.
        BufferedReader br_model = new BufferedReader(new FileReader(modelFileName));
        BufferedWriter bw_model_tmp = new BufferedWriter(new FileWriter(modelFileName + "_tmp"));

        bw_model_tmp.write(t_way_string);
        bw_model_tmp.newLine();

        String line;
        while ((line = br_model.readLine()) != null) {
            bw_model_tmp.write(line);
            bw_model_tmp.newLine();
        }
        br_model.close();
        bw_model_tmp.close();

        String[] command = {"java", "-jar", "ipog-ft.jar", modelFileName + "_tmp", constraintsFileName, t_way_string};
        Process process = Runtime.getRuntime().exec(command);

        InputStream istr = process.getInputStream();
        BufferedReader br = new BufferedReader(new InputStreamReader(istr));
        String str;
        int count = 0;
        String[] paraSequence_s = null;
        int[] paraSequence = new int[paraNum];
        ArrayList<String> initarray_s = new ArrayList<String>();
        // IPOG prints one additional constraint-related line when forbidden tuples
        // are present, which shifts the parameter-order and test-case offsets.
        while ((str = br.readLine()) != null) {
            count++;
            if (ftcount > 0) {
                if (count == 7)
                    paraSequence_s = str.split(" ");
                if (count >= 9) {
                    initarray_s.add(str);
                }
            } else {
                if (count == 6)
                    paraSequence_s = str.split(" ");
                if (count >= 8) {
                    initarray_s.add(str);
                }
            }
        }
        if (paraNum != paraSequence_s.length) {
            System.out.println("initialization fail!");
            System.exit(0);
        }
        for (int j = 0; j < paraNum; j++) {
            int value = Integer.parseInt(paraSequence_s[j]);
            paraSequence[j] = value;
        }

        int maxRowNum = initarray_s.size() - 1;
        coverageArray = new int[maxRowNum][paraNum];
        for (int i = 0; i < maxRowNum; i++) {
            str = initarray_s.get(i);
            String[] newStr = str.split(" ");
            if (newStr.length != paraNum) {
                System.out.println("initialization fail!");
                System.exit(0);
            }
            for (int j = 0; j < paraNum; j++) {
                int value = Integer.parseInt(newStr[j]);
                coverageArray[i][paraSequence[j]] = value;
            }
        }
        try {
            process.waitFor();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        br.close();

        File tmpFile = new File(modelFileName + "_tmp");
        if (tmpFile.exists())
            tmpFile.delete();

        return maxRowNum;
    }

}
