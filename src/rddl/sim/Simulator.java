/**
 * RDDL: Implements the RDDL simulator... see main().
 *
 * @author Scott Sanner (ssanner@gmail.com)
 * @version 10/10/10
 *
 **/

package rddl.sim;

import java.io.*;
import java.util.*;

import rddl.*;
import rddl.viz.*;
import rddl.viz.GenericScreenDisplay;
import rddl.solver.mdp.vi.VI;
import rddl.policy.*;
import rddl.RDDL.*;
import rddl.parser.parser;
import org.apache.commons.math3.*;
import org.apache.commons.math3.random.*;

import java.io.FileWriter;
import java.io.IOException;
import java.io.File;

public class Simulator {

	public State      _state;
	public INSTANCE   _i;
	public NONFLUENTS _n;
	public DOMAIN     _d;
	public Policy     _p;
	public StateViz   _v;
	public RandomDataGenerator _rand;


	public boolean started = false;
	public FileWriter fw;
	public StringBuilder sb;
	public boolean pomdp;

	public Simulator(RDDL rddl, String instance_name) throws Exception {
		_state = new State();
		this.pomdp = instance_name.contains("pomdp");
		// Set up instance, nonfluent, and domain information
		_i = rddl._tmInstanceNodes.get(instance_name);
		if (_i == null)
			throw new Exception("\nERROR: Instance '" + instance_name +
					"' not found, choices are " + rddl._tmInstanceNodes.keySet());
		_n = null;
		if (_i._sNonFluents != null) {
			_n = rddl._tmNonFluentNodes.get(_i._sNonFluents);
			if (_n == null)
				throw new Exception("\nERROR: Nonfluents '" + _i._sNonFluents +
						"' not found, choices are " + rddl._tmNonFluentNodes.keySet());
		}
		_d = rddl._tmDomainNodes.get(_i._sDomain);
		if (_n != null && !_i._sDomain.equals(_n._sDomain))
			throw new Exception("\nERROR: Domain name of instance and fluents do not match: " +
					_i._sDomain + " vs. " + _n._sDomain);
	}

	public void resetState() throws EvalException {
		_state.init(_d._hmObjects, _n != null ? _n._hmObjects : null, _i._hmObjects,
				_d._hmTypes, _d._hmPVariables, _d._hmCPF,
				_i._alInitState, _n == null ? new ArrayList<PVAR_INST_DEF>() : _n._alNonFluents, _i._alNonFluents,
				_d._alStateConstraints, _d._alActionPreconditions, _d._alStateInvariants,
				_d._exprReward, _i._nNonDefActions);
	}

	//////////////////////////////////////////////////////////////////////////////

	public Result run(Policy p, StateViz v, long rand_seed, int trial) throws EvalException {

		// Signal start of new session-independent round
		//p.roundInit(Double.MAX_VALUE, _i._nHorizon, 1, 1);

		// Reset to initial state
		resetState();

		// Set random seed for repeatability
		_rand = new RandomDataGenerator();
		_rand.reSeed(rand_seed);

		// Keep track of reward
		double accum_reward = 0.0d;
		double cur_discount = 1.0d;
		ArrayList<State> states = new ArrayList<State>(_i._nHorizon != Integer.MAX_VALUE ? _i._nHorizon : 1000);
		ArrayList<Double> rewards = new ArrayList<Double>(_i._nHorizon != Integer.MAX_VALUE ? _i._nHorizon : 1000);

		// Run problem for specified horizon
		for (int t = 0; t < _i._nHorizon; t++) {

			// Check state invariants to verify legal state -- can only reference current
			// state / derived fluents
			_state.checkStateInvariants();

			// Get action from policy
			// (if POMDP and first state, no observations available yet so a null is passed)
			State state_info = ((_state._alObservNames.size() > 0) && t == 0) ? null : _state;
			ArrayList<PVAR_INST_DEF> action_list = p.getActions(state_info);

			// Check action preconditions / state-action constraints (latter now deprecated)
			// (these constraints can mention actions and current state / derived fluents)
			_state.checkStateActionConstraints(action_list);

			// Compute next state (and all intermediate / observation variables)
			_state.computeNextState(action_list, _rand);

			// Display state/observations that the agent sees
			//v.display(_state, t);

			// Calculate reward / objective and store
			double reward = RDDL.ConvertToNumber(
					_state._reward.sample(new HashMap<LVAR,LCONST>(), _state, _rand)).doubleValue();
			rewards.add(reward);
			states.add(_state);
			accum_reward += cur_discount * reward;
			cur_discount *= _i._dDiscount;

			// Done with this iteration, advance to next round
			_state.advanceNextState(false /* do not clear observations */);

			// A "terminate-when" condition in the horizon specification may lead to early termination
			if (_i._termCond != null && _state.checkTerminationCondition(_i._termCond))
				break;
		}

		// Signal start of new session-independent round
		//p.roundEnd(accum_reward);

		// Problem over, return objective and list of rewards (e.g., for std error calc)
		v.close();
		write_data(states, rewards, trial);
		return new Result(accum_reward, rewards);
	}

	public void write_data(ArrayList<State> states, ArrayList<Double> rewards, int trial) {
		String fileName = "data_output.tsv";

		try {
		  File outFile = new File(fileName);
		  if (!outFile.exists()) {
				outFile.createNewFile();
				this.sb = new StringBuilder("");
				this.fw = new FileWriter(fileName, true);
		  } else if (!this.started) {
				outFile.delete();
				outFile.createNewFile();
				this.sb = new StringBuilder("");
				this.fw = new FileWriter(fileName, true);
		  } else {
				this.sb.append("\n");
			}
			this.started = true;
			this.sb.append(trial + "\t");
			for (int i=0; i<states.size(); i++){
				String stateStr = getStateDescription(states.get(i));
				if (i < states.size()-1){
		  		this.sb.append(stateStr + rewards.get(i) + "\t");
				} else {
					this.sb.append(stateStr + rewards.get(i));
				}
			}
			//this.sb.setLength(this.sb.length() - 1);
			if (trial%100==0) {
				System.out.println("trial:\t"+trial);
				this.fw.write(this.sb.toString());
				this.sb = new StringBuilder("");
			}
		} catch (IOException ioe) {
		  System.out.println("IOException occurred - " + ioe.getMessage());
		}
	}

	public String getStateDescription(State s) {
  	StringBuilder sb = new StringBuilder();

  	PVAR_NAME state = new PVAR_NAME("running");
  	PVAR_NAME obs   = new PVAR_NAME("running-obs");

		for (Map.Entry<String,ArrayList<PVAR_NAME>> e : s._hmTypeMap.entrySet()) {

			// Go through all variable names p for a variable type
			for (PVAR_NAME p : e.getValue()) {

				// Show interms only if they are derived
				PVARIABLE_DEF def = s._hmPVariables.get(p._pvarUnprimed);
				if (def instanceof PVARIABLE_INTERM_DEF
					&& !((PVARIABLE_INTERM_DEF)def)._bDerived)
					continue;
				String var_type = e.getKey();
				var_type = var_type.replace("interm", "derived");
				if (var_type.equals("nonfluent") || (this.pomdp && var_type.equals("states"))) continue;
		  	try {
		  	    // Go through all term groundings for variable p
		  	    ArrayList<ArrayList<LCONST>> gfluents = s.generateAtoms(p);
						System.out.println("\n- " + var_type + ": " + p);
		  	    for (ArrayList<LCONST> gfluent : gfluents){
							System.out.println("type:\t"+s.getPVariableAssign(p, gfluent).getClass());
							if ((s.getPVariableAssign(p, gfluent) instanceof Boolean)){
								sb.append(((Boolean)s.getPVariableAssign(p, gfluent) ? "1\t" : "0\t"));
								System.out.println(s.getPVariableAssign(p, gfluent));
							} else {
								sb.append(s.getPVariableAssign(p, gfluent));
								sb.append("\t");
								System.out.println(s.getPVariableAssign(p, gfluent));
							}
						}
						if (s._hmPVariables.get(obs) != null) {
				  		for (ArrayList<LCONST> gfluent : gfluents){
								if ((s.getPVariableAssign(obs, gfluent) instanceof Boolean)){
									sb.append(((Boolean)s.getPVariableAssign(obs, gfluent) ? "1\t" : "0\t"));
									System.out.println(s.getPVariableAssign(obs, gfluent));
								} else {
									sb.append(s.getPVariableAssign(obs, gfluent));
									sb.append("\t");
									System.out.println(s.getPVariableAssign(obs, gfluent));
								}
							}
	  	    	}
		  	} catch (EvalException ex) {
		  	    System.out.println("- could not retrieve assignment for " + p + "/" + obs + "\n");
		  	}
			}
		}
		return sb.toString();
	}

	//////////////////////////////////////////////////////////////////////////////

	public static void main(String[] args) throws Exception {

		// Argument handling
		if (args.length < 3 || args.length > 6) {
			System.out.println("usage: RDDL-file policy-class-name instance-name [state-viz-class-name] [#trials]");
			System.exit(1);
		}
		String rddl_file = args[0];
		String policy_class_name = args[1];
		String instance_name = args[2];
		String state_viz_class_name = "rddl.viz.GenericScreenDisplay";
		int trials = 1;
		if (args.length >= 4)
			state_viz_class_name = args[3];
		if (args.length >= 5)
			trials = new Integer(args[4]);

		// Load RDDL files
		RDDL rddl = new RDDL(rddl_file);
		Simulator sim = new Simulator(rddl, instance_name);

		for (int i=1; i<trials+1; i++){
			// Initialize simulator, policy and state visualization
			int rand_seed_sim = (int)System.currentTimeMillis(); // 123456
			int rand_seed_policy = (int)System.currentTimeMillis(); // 123456
			Policy pol = (Policy)Class.forName(policy_class_name).getConstructor(
					new Class[]{String.class}).newInstance(new Object[]{instance_name});
			pol.setRandSeed(rand_seed_policy);
			pol.setRDDL(rddl);

			//rddl.viz.GenericScreenDisplay gsd = new rddl.viz.GenericScreenDisplay();
			StateViz viz = (StateViz)Class.forName(state_viz_class_name).newInstance();
			//StateViz viz = (StateViz)gsd;

			// Reset, pass a policy, a visualization interface, a random seed, and simulate!
			Result r = sim.run(pol, viz, rand_seed_sim, i);
		}
		sim.fw.close();
	}
}
