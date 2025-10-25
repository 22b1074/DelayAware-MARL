import supersuit as ss

def make_env(scenario_name, discrete_action=False):
    from pettingzoo.mpe.simple_speaker_listener_v4 import parallel_env as ssl_env
    from pettingzoo.mpe.simple_spread_v3 import parallel_env as ss_env
    from pettingzoo.mpe.simple_reference_v3 import parallel_env as sr_env

    scenario_dict = {
        'simple_speaker_listener': ssl_env,
        'simple_spread': ss_env,
        'simple_reference': sr_env,
    }

    if scenario_name not in scenario_dict:
        raise ValueError(f"Scenario {scenario_name} not found in MPE2 environments")
    
    # Create the base environment
    base_env = scenario_dict[scenario_name](
        max_cycles=25,
        continuous_actions=not discrete_action
    )

    # Pad observations and actions for consistency
    env = ss.pad_observations_v0(base_env)
    env = ss.pad_action_space_v0(env)

    # Convert to vectorized environment (batch)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # Save agent info for DelayAware-MARL (needed later)
    env.num_agents = len(base_env.agents)
    env.agent_types = getattr(base_env, 'agent_types', ['agent'] * env.num_agents)
    env.single_env = base_env  # Keep access to single env

    return env
