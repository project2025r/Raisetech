# Role-Based Access Control (RBAC) Implementation

This document explains the Role-Based Access Control (RBAC) system implemented for the Road AI Safety Enhancement application.

## Overview

The RBAC system controls access to data on the Dashboard based on user roles. It ensures that users can only view data from users with equal or lower privileges in the role hierarchy.

## Role Hierarchy

The application supports three user roles with the following access permissions:

1. **Admin**: Can view all data uploaded by any role (Admin, Supervisor, Field Officer)
2. **Supervisor**: Can view data uploaded by Supervisors and Field Officers only
3. **Field Officer**: Can view data uploaded by Field Officers only

## Implementation Details

### Backend Components

#### 1. RBAC Utilities (`backend/utils/rbac.py`)

Core functions for role-based access control:

- `get_allowed_roles(user_role)`: Returns list of roles a user can access
- `create_role_filter(user_role)`: Creates MongoDB filter for role-based queries
- `validate_user_role(user_role)`: Validates if a role is valid
- `can_user_access_data(user_role, data_owner_role)`: Checks data access permissions

#### 2. Authentication Middleware (`backend/utils/auth_middleware.py`)

Provides decorators and utilities for API endpoint protection:

- `@validate_rbac_access`: Validates user role in requests
- `@require_role(allowed_roles)`: Restricts endpoint access to specific roles
- `get_current_user_role()`: Extracts user role from request
- `check_data_access_permission()`: Validates data access permissions

#### 3. Dashboard Routes (`backend/routes/dashboard.py`)

Updated to include RBAC filtering:
- All dashboard endpoints now accept `user_role` parameter
- Data is filtered based on user's role permissions
- MongoDB queries include role-based filters

#### 4. Users Routes (`backend/routes/users.py`)

Updated to filter user lists based on requesting user's role:
- `/api/users/all` endpoint filters users based on RBAC permissions
- Only shows users that the requesting user can access

### Frontend Components

#### 1. Dashboard Component (`frontend/src/pages/Dashboard.js`)

Updated to support RBAC:
- Receives user information as props
- Passes user role to all API calls
- Filters user dropdown based on role permissions
- All data displayed respects role-based access control

#### 2. App Component (`frontend/src/App.js`)

Updated to pass user information to Dashboard component.

## API Usage

### Request Format

All dashboard API endpoints now accept an optional `user_role` parameter:

```javascript
// GET request with user role
axios.get('/api/dashboard/summary', {
  params: {
    user_role: 'Admin',
    start_date: '2025-01-01',
    end_date: '2025-01-31'
  }
})
```

### Response Filtering

Data in responses is automatically filtered based on the user's role:

- **Admin**: Sees all data from all users
- **Supervisor**: Sees data from Supervisors and Field Officers only
- **Field Officer**: Sees data from Field Officers only

## Testing

### Manual Testing

1. **Test as Admin**:
   - Login as Admin user
   - Navigate to Dashboard
   - Verify you can see data from all user types
   - Check user filter dropdown includes all users

2. **Test as Supervisor**:
   - Login as Supervisor user
   - Navigate to Dashboard
   - Verify you only see data from Supervisors and Field Officers
   - Check user filter dropdown excludes Admin users

3. **Test as Field Officer**:
   - Login as Field Officer user
   - Navigate to Dashboard
   - Verify you only see data from Field Officers
   - Check user filter dropdown only shows Field Officer users

### Automated Testing

Run the RBAC test script:

```bash
cd backend
python test_rbac.py
```

This will test:
- Role validation
- Allowed roles for each user type
- MongoDB filter creation
- Data access permissions
- Real-world scenarios

## Security Considerations

### Backend Validation

- All role validations happen on the backend
- Frontend role filtering is supplementary (UX improvement)
- API endpoints validate user roles before processing requests
- Invalid roles are rejected with appropriate error messages

### Data Protection

- Users cannot access data they don't have permission to view
- Role hierarchy is enforced at the database query level
- No sensitive data is exposed to unauthorized users

## Error Handling

The system handles various error scenarios:

- **Invalid Role**: Returns 400 Bad Request
- **Missing Role**: Continues with no filtering (backward compatibility)
- **Insufficient Permissions**: Returns 403 Forbidden
- **Database Errors**: Returns 500 Internal Server Error

## Backward Compatibility

The RBAC system is designed to be backward compatible:
- If no user role is provided, the system behaves as before
- Existing API calls continue to work without modification
- Frontend gracefully handles missing user information

## Future Enhancements

Potential improvements for the RBAC system:

1. **Role-Based UI**: Hide/show UI elements based on user role
2. **Audit Logging**: Log all data access attempts
3. **Dynamic Permissions**: Configure role permissions via admin interface
4. **Session Management**: Implement proper session-based authentication
5. **API Rate Limiting**: Implement rate limiting per role

## Troubleshooting

### Common Issues

1. **Data Not Showing**: Check user role is being passed correctly
2. **Permission Errors**: Verify user has correct role in database
3. **Filter Not Working**: Ensure role parameter is included in API calls

### Debug Steps

1. Check browser console for API errors
2. Verify user role in sessionStorage
3. Check backend logs for role validation errors
4. Test with different user roles

## Configuration

### Adding New Roles

To add new roles:

1. Update `valid_roles` list in `utils/rbac.py`
2. Update `role_hierarchy` in `get_allowed_roles()` function
3. Update frontend role handling if needed
4. Update authentication system to support new roles

### Modifying Permissions

To modify role permissions:

1. Update `role_hierarchy` in `get_allowed_roles()` function
2. Test thoroughly with all affected roles
3. Update documentation

## Conclusion

The RBAC system provides secure, role-based access control for the Dashboard while maintaining backward compatibility and ease of use. It ensures that users only see data they are authorized to view, implementing the principle of least privilege. 