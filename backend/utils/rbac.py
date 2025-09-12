def get_allowed_roles(user_role):
    """
    Determine which roles' data a user can view based on their role.
    
    Args:
        user_role (str): The role of the requesting user
        
    Returns:
        list: List of roles whose data the user can view
    """
    role_hierarchy = {
        'Admin': ['Admin', 'Supervisor', 'Inspector'],
        'Supervisor': ['Supervisor', 'Inspector'],
        'Inspector': ['Inspector']
    
    }
    
    return role_hierarchy.get(user_role, ['Inspector'])  # Default to Inspector if role not found

def create_role_filter(user_role):
    """
    Create a MongoDB filter for role-based access control.
    
    Args:
        user_role (str): The role of the requesting user
        
    Returns:
        dict: MongoDB filter for allowed roles
    """
    allowed_roles = get_allowed_roles(user_role)
    
    if len(allowed_roles) == 1:
        return {"role": allowed_roles[0]}
    else:
        return {"role": {"$in": allowed_roles}}

def validate_user_role(user_role):
    """
    Validate if the provided user role is valid.
    
    Args:
        user_role (str): The role to validate
        
    Returns:
        bool: True if role is valid, False otherwise
    """
    valid_roles = ['Admin', 'Supervisor', 'Inspector']
    return user_role in valid_roles

def can_user_access_data(user_role, data_owner_role):
    """
    Check if a user can access data uploaded by another user.
    
    Args:
        user_role (str): The role of the requesting user
        data_owner_role (str): The role of the user who uploaded the data
        
    Returns:
        bool: True if user can access the data, False otherwise
    """
    allowed_roles = get_allowed_roles(user_role)
    return data_owner_role in allowed_roles 