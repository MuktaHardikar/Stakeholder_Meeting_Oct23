from pyomo.environ import ConcreteModel, Var, Objective, units as pyunits
from idaes.core import FlowsheetBlock
import pandas as pd
from steady_state_flowsheets.battery import BatteryStorage
from steady_state_flowsheets.simple_RO_unit import ROUnit
import datetime

def define_system_vars(m):
    if "USD_2021" not in pyunits._pint_registry:
            pyunits.load_definitions_from_strings(
                ["USD_2021 = 500/708.0 * USD_CE500"]
            )

    m.fs.pv_to_ro = Var(
            initialize = 100,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'PV to RO electricity'
        )
    m.fs.grid_to_ro = Var(
            initialize = 100,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'Grid to RO electricity'
        )
    m.fs.curtailment = Var(
            initialize = 0,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'PV curtailment'
        )

    m.fs.elec_price = Var(
            initialize = 0.1,
            bounds = (0,None),
            units = pyunits.USD_2021,
            doc = 'Electric Cost'
        )

    m.fs.elec_generation = Var(
            initialize = 1000,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'PV Power Gen'
        )

    m.fs.pv_gen = Var(
            initialize = 1000,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'PV Power Gen'
        )

    m.fs.ro_elec_req = Var(
            initialize = 1000,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'RO Power Demand'
        )
        
    m.fs.electricity_price = Var(
            initialize = 0.1,
            bounds = (0,None),
            units = pyunits.USD_2021,
            doc = 'Electricity Price'
        )

def add_steady_state_constraints(m):
    # Add energy flow balance
    @m.Constraint(doc="System energy flow")
    def eq_pv_elec_gen(b):
        return (
        m.fs.pv_gen == b.fs.pv_to_ro + b.fs.battery.elec_in[0] + b.fs.curtailment
        )

    @m.Constraint(doc="RO electricity requirment")
    def eq_ro_elec_req(b):
        return (m.fs.RO.power_demand == b.fs.pv_to_ro + b.fs.battery.elec_out[0] + b.fs.grid_to_ro
        )

    # Add grid electricity cost
    @m.Expression(doc="grid cost")
    def grid_cost(b):
        return (m.fs.electricity_price * b.fs.grid_to_ro * b.fs.battery.dt)

def add_pv_ro_constraints(mp):
    """
    This function adds constraints that connect variables across two time periods

    Args:
        m: Pyomo model

    Returns:
        None
    """

    cost_battery_power = 75 # $/kW
    cost_battery_energy = 50 # $/kWh  
    n_time_points= 24
    ro_capacity = 6000 # m3/day

    @mp.Expression(doc="battery cost")
    def battery_capital_cost(b):
        return ((cost_battery_power * b.blocks[0].process.fs.battery.nameplate_power +
                 cost_battery_energy * b.blocks[0].process.fs.battery.nameplate_energy))
        
    # Add PV cost function
    @mp.Expression(doc="PV cost")
    def pv_capital_cost(b):
        return (0.37 * 1000 * b.blocks[0].process.fs.pv_size +
                0.03 * 1000 * b.blocks[0].process.fs.pv_size)

    @mp.Expression(doc="Capital cost")
    def total_capital_cost(b):
        return (b.battery_capital_cost + b.pv_capital_cost)
    
    @mp.Expression(doc="Annualized Capital cost")
    def annualized_capital_cost(b):
        return (b.total_capital_cost / 20)

    # Total cost
    @mp.Expression(doc='total cost')
    def total_cost(b):
        # The annualized capital cost is evenly distributed to the multiperiod
        return (
            (b.annualized_capital_cost) / 365 / 24 * n_time_points
            + sum([b.blocks[i].process.grid_cost for i in range(n_time_points)])
        )

    # LCOW
    @mp.Expression(doc='total cost')
    def LCOW(b):
        # LCOW from RO: 0.45
        return (
            0.40 + b.total_cost / (n_time_points*pyunits.convert(ro_capacity * pyunits.m**3 / pyunits.day, to_units=pyunits.m**3 / pyunits.hour))
        )   

    # Set objective
    mp.obj = Objective(expr=mp.LCOW)

def eval_surrogate(surrogate, design_size = 1000,Day = 1, Hour = 1):
    input = pd.DataFrame.from_dict([{'design_size':design_size, 'Hour':Hour, 'Day':Day}], orient='columns')
    hourly_gen = surrogate.evaluate_surrogate(input)
    return max(0,hourly_gen.values[0][0])

def unfix_dof(m):
    """
    This function unfixes a few degrees of freedom for optimization

    Args:
        m: object containing the integrated nuclear plant flowsheet

    Returns:
        None
    """
    # m.fs.battery.nameplate_energy.unfix()
    # m.fs.battery.nameplate_power.unfix()
    m.fs.battery.nameplate_energy.fix(8000)
    m.fs.battery.nameplate_power.fix(400)

def fix_dof_and_initialize(
    m,
):
    """Fix degrees of freedom and initialize the flowsheet

    This function fixes the degrees of freedom of each unit and initializes the entire flowsheet.

    Args:
        m: Pyomo `Block` or `ConcreteModel` containing the flowsheet
        outlvl: Logger (default: idaeslog.WARNING)
    """
    m.fs.battery.initialize()

    return 

def add_battery(m):
    """ This model does not use the flowsheet's time domain. Instead, it only models a single timestep, with initial
    conditions provided by `initial_state_of_charge` and `initial_energy_throughput`. The model calculates change
    in stored energy across a single time step using the power flow variables, `power_in` and `power_out`, and
    the `dr_hr` parameter."""

    m.fs.battery = BatteryStorage()
    m.fs.battery.charging_eta.set_value(0.95)    # Charging efficiency
    m.fs.battery.discharging_eta.set_value(0.95) # Discharging efficiency
    m.fs.battery.dt.set_value(1)                 # Time step

    return m.fs.battery

def steady_state_flowsheet(m = None,
                               pv_gen = 1000,
                               electricity_price = 0.1,
                               ro_capacity = 6000,
                               ro_elec_req = 944.3,
                               pv_oversize = 1,
                               fixed_battery_size = None):
    
    if m is None:
        m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.pv_size = pv_oversize*ro_elec_req
    m.fs.battery = add_battery(m)

    m.fs.RO = ROUnit()
    define_system_vars(m)
    add_steady_state_constraints(m)
    m.fs.pv_gen.fix(pv_gen)
    m.fs.electricity_price.fix(electricity_price)
    m.fs.elec_price.fix(electricity_price)

    return m

def get_elec_tier(Hour = 1):
    electric_tiers = {'Tier 1':0.19825,'Tier 2':0.06124,'Tier 3':0.24445,'Tier 4':0.06126}
    if (Hour < 12) | (Hour > 18):
        return electric_tiers['Tier 2']
    else:
        return electric_tiers['Tier 1']