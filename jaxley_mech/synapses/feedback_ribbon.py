from typing import Optional

import jax.numpy as jnp
from jaxley.solver_gate import save_exp
from jaxley.synapses.synapse import Synapse

from jaxley_mech.solvers import SolverExtension


class FeedbackRibbon(Synapse, SolverExtension):
    def __init__(
        self,
        name: Optional[str] = None,
        solver: Optional[str] = None,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 10,
    ):
        super().__init__(name)
        SolverExtension.__init__(self, solver, rtol, atol, max_steps)
        self.prefix = self._name

        self.synapse_params = {
            # V cone to iCa2+
            f"{self.prefix}_VGCC_gmax": 1.0507,  # nS
            f"{self.prefix}_VGCC_Vhalf": -38.8631,  # mV
            f"{self.prefix}_VGCC_k": 4.6478,  # mV
            f"{self.prefix}_VGCC_Erev": 9.5288,  # mV
            f"{self.prefix}_VGCC_Shift": -15.5,  # mV/pH unit (Barnes and Bui, 1991)
            f"{self.prefix}_VGCC_Gain": 0.71,  # pH^-1
            # iCa2+ to [Ca2+]
            f"{self.prefix}_iCaToCa": 1e-7,  # a.u./pA
            f"{self.prefix}_Ca_tau": 50,
            # [Ca] -> [Glu] release
            f"{self.prefix}_e_max": 1.5,  # Maximum glutamate release
            f"{self.prefix}_r_max": 2.0,  # Rate of RP --> IP, movement to the ribbon
            f"{self.prefix}_i_max": 4.0,  # Rate of IP --> RRP, movement to the dock
            f"{self.prefix}_d_max": 0.1,  # Rate of RP refilling
            f"{self.prefix}_RRP_max": 3.0,  # Maximum number of docked vesicles
            f"{self.prefix}_IP_max": 10.0,  # Maximum number of vesicles at the ribbon
            f"{self.prefix}_RP_max": 25.0,  # Maximum number of vesicles in the reserve pool
            f"{self.prefix}_Ca_half": 1e-4,  # Half the [Ca] that gives maximum glutamate release
            # Glu -> iGlu
            f"{self.prefix}_G_syn": 1e-04,  # uS
            f"{self.prefix}_Erev_syn": 0.0,  # mV
            f"{self.prefix}_pH_Tau": 5.0,
            f"{self.prefix}_pH_Base": 7.4,  # standard pH  value
            f"{self.prefix}_pH_Base_active": 1.0,  # 1 = active, 0 = inactive
            # Ephatic
            f"{self.prefix}_Ephaptic_C": 1.8e-1,
            f"{self.prefix}_Ephaptic_Erev": -35.0,  # mV
            f"{self.prefix}_Ephaptic_active": 1.0,  # 1 = active, 0 = inactive
            # SLC4A5
            f"{self.prefix}_SLC4A5_C": 8e-11,
            f"{self.prefix}_SLC4A5_Erev": -130,  # mV
            f"{self.prefix}_SLC4A5_active": 1.0,  # 1 = active, 0 = inactive
            # NHE
            f"{self.prefix}_NHE_C": 5e-10,  # pH unit/mV
            f"{self.prefix}_NHE_Offset": -70,  # mV
            f"{self.prefix}_NHE_k": 15,  # mV
            f"{self.prefix}_NHE_active": 1.0,  # 1 = active, 0 = inactive
            # Panx1
            f"{self.prefix}_Panx1_EHalfOpen": -55,  # mV
            f"{self.prefix}_Panx1_Tau": 2e2,
            f"{self.prefix}_Panx1_C": 1,
            f"{self.prefix}_Panx1_pKs": 7.2,  # pH unit
            f"{self.prefix}_Panx1_k": 3.0,  # mV
            f"{self.prefix}_Panx1_active": 1.0,  # 1 = active, 0 = inactive
            # GABA
            f"{self.prefix}_GABA_EHalfOpen": -55,  # mV
            f"{self.prefix}_GABA_Tau": 2e4,
            f"{self.prefix}_GABA_C_I": 5e-1,  # nS
            f"{self.prefix}_GABA_C_pH": 1e-9,
            f"{self.prefix}_GABA_Erev_HCO3": -20,  # mV
            f"{self.prefix}_GABA_Erev_Cl": -28,  # mV
            f"{self.prefix}_GABA_active": 1.0,  # 1 = active, 0 = inactive
            f"{self.prefix}_feedback": 1.0,  # 1 = active, 0 = inactive (for all feedback mechanisms)
        }

        self.synapse_states = {
            f"{self.prefix}_VGCC_i": 0.0,  # Calcium current (pA) into presynaptic photoreceptor
            f"{self.prefix}_Ca": 1.0,  # Presynaptic intracellular [Ca] (a.u.)
            f"{self.prefix}_exo": 0.75,  # Number of vesicles released
            f"{self.prefix}_RRP": 1.5,  # Number of vesicles at the dock
            f"{self.prefix}_IP": 5.0,  # Number of vesicles at the ribbon
            f"{self.prefix}_RP": 12.5,  # Number of vesicles in the reserve pool
            f"{self.prefix}_pH": 7.4,  # Synaptic cleft pH
            f"{self.prefix}_Panx1_OpenProb": 1.0,  # Panx1 channels open probability
            f"{self.prefix}_GABA_OpenProb": 0.0,  # GABA channels open probability
            f"{self.prefix}_vShiftPH": 0.0,  # pH induced voltage shift (mV)
            f"{self.prefix}_vShiftEph": 0.0,  # Ephaptic induced voltage shift (mV) (equal to synaptic cleft potential)
            f"{self.prefix}_vEff": 0.0,  # Effective voltage (mV) sensed by VGCCs
            f"{self.prefix}_iGlu": 0.0,  # Glutamte induced postsynaptic current into HC
            f"{self.prefix}_iGABA": 0.0,  # GABA induced postsynaptic current into HC
        }

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""

        Ca = states[f"{self.prefix}_Ca"]
        pH = states[f"{self.prefix}_pH"]
        Panx1_OpenProb = states[f"{self.prefix}_Panx1_OpenProb"]
        GABA_OpenProb = states[f"{self.prefix}_GABA_OpenProb"]
        VShiftEph = states[f"{self.prefix}_vShiftEph"]

        # update VShiftEph
        VShiftEph = (
            (post_voltage - params[f"{self.prefix}_Ephaptic_Erev"])
            * params[f"{self.prefix}_Ephaptic_C"]
            * params[f"{self.prefix}_Ephaptic_active"]
            * params[f"{self.prefix}_feedback"]
        )

        # update Panx1 open probability
        Panx1_OpenProb = self.vHor_to_Panx1OpenProb(
            Panx1_OpenProb, post_voltage, params, delta_t
        )

        # update GABA open probability
        GABA_OpenProb = self.vHor_to_GABAOpenProb(
            GABA_OpenProb, post_voltage, params, delta_t
        )

        # calculate pH
        pH = self.vHor_to_pH(
            pH, post_voltage, Panx1_OpenProb, GABA_OpenProb, params, delta_t
        )

        # calculate VShiftPH
        VShiftPH = (
            params[f"{self.prefix}_VGCC_Shift"]
            * (pH - params[f"{self.prefix}_pH_Base"])
            * params[f"{self.prefix}_feedback"]
        )
        IpeakShiftpH = 1 + (
            params[f"{self.prefix}_VGCC_Gain"]
            * (pH - params[f"{self.prefix}_pH_Base"])
            * params[f"{self.prefix}_feedback"]
        )

        # update vEff
        vEff = pre_voltage - VShiftEph - VShiftPH

        # Vcone to Ca2+ current (Grove, 2019)
        iCa = self.vCone_to_iCa(vEff, IpeakShiftpH, params)

        Ca = self.iCa_to_ca(params, iCa, Ca, delta_t)

        # glutamate release
        exo, rrp, ip, rp = (
            states[f"{self.prefix}_exo"],
            states[f"{self.prefix}_RRP"],
            states[f"{self.prefix}_IP"],
            states[f"{self.prefix}_RP"],
        )
        exo, rrp, ip, rp = self.ca_to_glu(params, Ca, exo, rrp, ip, rp, delta_t)

        # glutamate induced current (iGlu)
        iGlu = (
            params[f"{self.prefix}_G_syn"]
            * states[f"{self.prefix}_exo"]
            * (post_voltage - params[f"{self.prefix}_Erev_syn"])
            * 1e3
        )  # mV * uS * 1e3 = nA * 1e3 = pA
        # GABA induced Cl- current (iGABA)  (nS * mV = pA)
        iGABA = (
            params[f"{self.prefix}_feedback"]
            * params[f"{self.prefix}_GABA_active"]
            * params[f"{self.prefix}_GABA_C_I"]
            * states[f"{self.prefix}_GABA_OpenProb"]
            * (post_voltage - params[f"{self.prefix}_GABA_Erev_Cl"])
        )

        return {
            f"{self.prefix}_Ca": Ca,
            f"{self.prefix}_VGCC_i": iCa,
            f"{self.prefix}_pH": pH,
            f"{self.prefix}_exo": exo,
            f"{self.prefix}_RRP": rrp,
            f"{self.prefix}_IP": ip,
            f"{self.prefix}_RP": rp,
            f"{self.prefix}_iGlu": iGlu,
            f"{self.prefix}_iGABA": iGABA,
            f"{self.prefix}_Panx1_OpenProb": Panx1_OpenProb,
            f"{self.prefix}_GABA_OpenProb": GABA_OpenProb,
            f"{self.prefix}_vShiftEph": VShiftEph,
            f"{self.prefix}_vEff": vEff,
            f"{self.prefix}_vShiftPH": VShiftPH,
        }

    def vHor_to_Panx1OpenProb(self, Panx1_OpenProb, post_voltage, params, dt):
        """Update Panx1 channel opening probability.

        Steady state probability depends on horizontal cell voltage and is modeled using a logistic function.
        The steady state value is exponentially approached with a first order linear DEQ.
        """
        args_tuple = (
            params[f"{self.prefix}_Panx1_Tau"],
            params[f"{self.prefix}_Panx1_EHalfOpen"],
            params[f"{self.prefix}_Panx1_k"],
        )

        def derivative_Panx1(t, states, args):
            Panx1_OpenProb, post_voltage = states
            tau, EHalfOpen, k = args
            target = 1.0 / (1.0 + jnp.exp(-(post_voltage - EHalfOpen) / k))

            return -1 / tau * (Panx1_OpenProb - target)

        y0 = jnp.array([Panx1_OpenProb, post_voltage])
        y_Panx1 = self.solver_func(y0, dt, derivative_Panx1, args_tuple)
        return jnp.array(y_Panx1[0])

    def vHor_to_GABAOpenProb(self, GABA_OpenProb, post_voltage, params, dt):
        """Update GABA channel opening probability.

        Steady state probability depends on horizontal cell voltage and is modeled using a logistic function.
        The steady state value is exponentially approached with a first order linear DEQ.
        """
        args_tuple = (
            params[f"{self.prefix}_GABA_Tau"],
            params[f"{self.prefix}_GABA_EHalfOpen"],
        )

        def derivative_GABA(t, states, args):
            GABA_OpenProb, post_voltage = states
            tau, EHalfOpen = args
            target = 1.0 / (1.0 + jnp.exp(-(post_voltage - EHalfOpen) / 3))

            return -1 / tau * (GABA_OpenProb - target)

        y0 = jnp.array([GABA_OpenProb, post_voltage])
        y_GABA = self.solver_func(y0, dt, derivative_GABA, args_tuple)
        return jnp.array(y_GABA[0])

    def vHor_to_pH(self, pH, post_voltage, Panx1_OpenProb, GABA_OpenProb, params, dt):
        """Update synaptic cleft pH.


        pH is model by summing the contributions of the different feedback mechanisms.
        Slc4a5:
        Warren et al., “Sources of Protons and a Role for Bicarbonate in Inhibitory Feedback from Horizontal Cells to Cones in Ambystoma Tigrinum Retina.”
        Morikawa et al., “The Sodium-Bicarbonate Cotransporter Slc4a5 Mediates Feedback at the First Synapse of Vision.”
        NHE:
        Barnes et al., “Horizontal Cell Feedback to Cone Photoreceptors in Mammalian Retina: Novel Insights From the GABA-pH Hybrid Model.”
        Panx1:
        Cenedese et al., “Pannexin 1 Is Critically Involved in Feedback from Horizontal Cells to Cones.”
        GABA:
        Barnes et al., “Horizontal Cell Feedback to Cone Photoreceptors in Mammalian Retina: Novel Insights From the GABA-pH Hybrid Model.”
        """
        args_tuple = (
            params[f"{self.prefix}_pH_Base"],
            params[f"{self.prefix}_pH_Tau"],
            params[f"{self.prefix}_pH_Base_active"],
            params[f"{self.prefix}_feedback"],
            params[f"{self.prefix}_SLC4A5_C"],
            params[f"{self.prefix}_SLC4A5_Erev"],
            params[f"{self.prefix}_SLC4A5_active"],
            params[f"{self.prefix}_NHE_C"],
            params[f"{self.prefix}_NHE_Offset"],
            params[f"{self.prefix}_NHE_k"],
            params[f"{self.prefix}_NHE_active"],
            params[f"{self.prefix}_Panx1_C"],
            params[f"{self.prefix}_Panx1_pKs"],
            Panx1_OpenProb,
            params[f"{self.prefix}_Panx1_active"],
            params[f"{self.prefix}_GABA_C_pH"],
            params[f"{self.prefix}_GABA_Erev_HCO3"],
            GABA_OpenProb,
            params[f"{self.prefix}_GABA_active"],
        )

        def derivative_pH(t, states, args):
            pH = states
            (
                target,
                tau,
                base_active,
                feedback_active,
                SLC4A5_C,
                SLC4A5_Erev,
                SLC4A45_active,
                NHE_C,
                NHE_Offset,
                NHE_K,
                NHE_active,
                Panx1_C,
                Panx1_pKs,
                Panx1_open,
                Panx1_active,
                GABA_C,
                GABA_Erev,
                GABA_OpenProb,
                GABA_active,
            ) = args

            dpH_Base = -base_active * 1 / tau * (pH - target)

            dpH_SLC4A5 = -SLC4A45_active * (
                SLC4A5_C * (post_voltage - SLC4A5_Erev) * jnp.pow(10, 14 - pH)
            )

            dpH_NHE = -NHE_active * (
                NHE_C * jnp.exp((post_voltage - NHE_Offset) / NHE_K) * jnp.pow(10, pH)
            )

            dpH_Panx1 = -Panx1_active * Panx1_open * (Panx1_C * (pH - Panx1_pKs))

            dpHGABA = (
                -GABA_active
                * GABA_OpenProb
                * (GABA_C * (post_voltage - GABA_Erev) * jnp.pow(10, 14 - pH))
            )

            dpH = (
                dpH_Base + dpH_SLC4A5 + dpH_NHE + dpH_Panx1 + dpHGABA
            ) * feedback_active

            return jnp.array([dpH])

        y0 = jnp.array([pH])
        y_pH = self.solver_func(y0, dt, derivative_pH, args_tuple)

        return jnp.array(y_pH[0])

    def vCone_to_iCa(self, vEff, peakGain, params):
        """Convert sensed potential by VGCCs into a cone calcium current."""
        gmax = params[f"{self.prefix}_VGCC_gmax"]
        Vhalf = params[f"{self.prefix}_VGCC_Vhalf"]
        k = params[f"{self.prefix}_VGCC_k"]
        Erev = params[f"{self.prefix}_VGCC_Erev"]

        return (
            gmax
            * (1.0 / (1.0 + save_exp(-(vEff - Vhalf) / k)))
            * (vEff - Erev)
            * peakGain
        )

    def iCa_to_ca(self, params, iCa, Ca, dt):
        """Convert cone calcium current into intracellular calcium concentration."""
        args_tuple = (
            params[f"{self.prefix}_iCaToCa"],
            params[f"{self.prefix}_Ca_tau"],
        )
        y0 = jnp.array([Ca, iCa])

        def derivative_Ca(t, states, args):
            Ca, iCa = states
            iCaToCa, decayCa = args

            return -1 / decayCa * Ca - iCa * iCaToCa

        y_new = self.solver_func(y0, dt, derivative_Ca, args_tuple)
        return y_new[0]

    def ca_to_glu(self, params, ca, exo, rrp, ip, rp, dt):
        """Convert cone intracellular calcium concentration to glutamate release. (from RibbonSynapse)"""
        args_tuple = (
            params[f"{self.prefix}_e_max"],
            params[f"{self.prefix}_r_max"],
            params[f"{self.prefix}_i_max"],
            params[f"{self.prefix}_d_max"],
            params[f"{self.prefix}_RRP_max"],
            params[f"{self.prefix}_IP_max"],
            params[f"{self.prefix}_RP_max"],
            params[f"{self.prefix}_Ca_half"],
            ca,
        )
        y0 = jnp.array([exo, rrp, ip, rp])

        def derivatives_ribon(t, states, args):
            exo, RRP, IP, RP = states
            e_max, r_max, i_max, d_max, RRP_max, IP_max, RP_max, Ca_half, ca = args

            # Presynaptic voltage to calcium to release probability
            p_d_t = 1.0 / (1.0 + save_exp(-1 * 3e4 * (ca - Ca_half)))

            # Glutamate release
            e_t = e_max * p_d_t * RRP / RRP_max
            # Rate of RP --> IP, movement to the ribbon
            r_t = r_max * (1 - IP / IP_max) * RP / RP_max
            # Rate of IP --> RRP, movement to the dock
            i_t = i_max * (1 - RRP / RRP_max) * IP / IP_max
            # Rate of RP refilling
            d_t = d_max * exo

            dRP_dt = d_t - r_t
            dIP_dt = r_t - i_t
            dRRP_dt = i_t - e_t
            dExo_dt = e_t - d_t

            return jnp.array([dExo_dt, dRRP_dt, dIP_dt, dRP_dt])

        y_new = self.solver_func(y0, dt, derivatives_ribon, args_tuple)

        return y_new

    def compute_current(self, states, pre_voltage, post_voltage, params):
        iGlu = states[f"{self.prefix}_iGlu"] * 1e-3  # pA -> nA
        iGABA = states[f"{self.prefix}_iGABA"] * 1e-3  # pA -> nA
        return iGlu + iGABA

    def reset_all_feedbacks(self, pHBase_active=True):
        self.synapse_states[f"{self.prefix}_Ephaptic_active"] = 0.0
        self.synapse_states[f"{self.prefix}_SLC4A5_active"] = 0.0
        self.synapse_states[f"{self.prefix}_NHE_active"] = 0.0
        self.synapse_states[f"{self.prefix}_Panx1_active"] = 0.0
        self.synapse_states[f"{self.prefix}_GABA_active"] = 0.0
        self.synapse_states[f"{self.prefix}_pH_Base_active"] = 1.0 if pHBase_active else 0.0
