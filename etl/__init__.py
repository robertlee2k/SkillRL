"""
ETL Pipeline for Customer Service RL Data

Converts raw chat logs to RL-ready Playbook JSON trees.
"""

from .config import (
    ROLE_MAPPING,
    SKILL_DEFINITIONS,
    STATE_MACHINE,
    VALID_SKILLS,
    SCENE_CATEGORIES,
    SCENE_PATTERNS
)
from .parser import parse_system_message, extract_slot_updates, SYSTEM_SLOT_PATTERNS
from .aggregator import aggregate_turns
from .cleaner import clean_session, validate_user_agent_alternation
from .batch import load_sessions, save_playbooks, process_batch
from .classifier import classify_scene
from .validator import validate_playbook, ValidationError, validate_playbook_with_details
from .pipeline import run_pipeline, build_playbook
from .llm_generator import call_llm_for_classification, call_llm_for_playbook, LLMGenerator
from .customer_service_env import CustomerServiceEnv, EnvState, create_env, run_random_episode

__all__ = [
    # Config
    'ROLE_MAPPING',
    'SKILL_DEFINITIONS',
    'STATE_MACHINE',
    'VALID_SKILLS',
    'SCENE_CATEGORIES',
    'SCENE_PATTERNS',
    # Parser
    'parse_system_message',
    'extract_slot_updates',
    'SYSTEM_SLOT_PATTERNS',
    # Aggregator
    'aggregate_turns',
    # Cleaner
    'clean_session',
    'validate_user_agent_alternation',
    # Batch
    'load_sessions',
    'save_playbooks',
    'process_batch',
    # Classifier
    'classify_scene',
    # Validator
    'validate_playbook',
    'ValidationError',
    'validate_playbook_with_details',
    # Pipeline
    'run_pipeline',
    'build_playbook',
    # LLM Generator
    'call_llm_for_classification',
    'call_llm_for_playbook',
    'LLMGenerator',
    # RL Environment
    'CustomerServiceEnv',
    'EnvState',
    'create_env',
    'run_random_episode',
]