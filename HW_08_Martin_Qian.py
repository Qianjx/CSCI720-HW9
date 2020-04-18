import math
import matplotlib.pyplot as plt
    
# All initial parameters
INITIAL_THETA           = 58 
INITIAL_RHO             = 10.75 
INITIAL_ALPHA           = 4.5 
MINIMUM_ALPHA_DELTA     = 0.01 
LEARNING_RATE           = 0.9 

average_dist_list = []
def Gradient_Descent__Fit_Through_a_Line_v100(xys, theta, rho, alpha ):
# Given a set of points, find a line through them, usin(math.radians(g gradient decent.

# We model the line we seek as:
#
# A * x + B * y + C = 0            
# This is the cannonical form, used in linear algebra and many other places.
#
# In our case we are going to use:
# A * x + B * y - rho = 0 
#
# We do this because, in a second we are going to constrain A and B in such 
# a way that rho becomes the distance from the origin to the line.
#
# We need to find A, B, and rho.
#
# However, this is three parameters.  
# For a line, we know we only need two parameters here.
# (For example, slope and intercept when usin(math.radians(g the form y = mx + b.
#
# So, the form   A * x + B * y - rho = 0 
# has 1 too many parameters.
#
# To simplify things, we will assume that 
# A = cos(math.radians((theta), and B = sin(math.radians((theta).
# 
# These are called the "directed cos(math.radians(ines" of a line.  
# They give the unit vectors for the line,
# based on the angle that the lines goes in.
#
# Then A and B are tied together, and the only two parameters we need are theta, and rho.
#
# So, writing this out again, the model of the line we have is:
# 
# cos(math.radians(ine(theta) * x + sin(math.radians(e(theta) * y - rho = 0 
#
# The angle, theta, it turns out, is the angle from the origin
# to the closest point on the line.  
#
# Now, IF ONLY WE COULD FIND THE BEST VALUES OF theta AND rho!! 
#
#
# Okay, so anytime someone says, "best" we have do ask the question, 
# "what do we mean by best?"
# Or, "Best How?"
#
# In our case, we want the best line to be the one that minimizes the total distance from all of 
# the points, to the line we want.  That gives the best line -- the one that minimizes the distance 
# from all of the points to the line.  OR, sin(math.radians(ce there are 6 points, a fixed number of points,
# we could minimize the average absolute distance.
#
# We know from lecture 5a&b (or so) that the distance from a point (x,y) to the line
# Ax + By + C = 0 is:
# 
# dst = abs( Ax + By + C ) / sqrt( A^2 + B^2 )
#
# In the case of each and every point:
#   A = cos(math.radians((theta)
#   B = sin(math.radians((theta)
#   C = rho
#
#
# Procedure:  Guess and Adjust
# Step 1.   Fix values of theta and rho.
#
# Step 2.   Find the distance of all the points to the line.
#
# Step 3A.  Try values of rho = rho +/- some change, alpha.
#           If the change decreases the total distance of the points to the line,
#           then keep making that change.
#
# Step 3B.  Try values of theta = theta +/- some change, alpha.
#           If the change decreases the total distance of the points to the line,
#           then keep making that change.
#
# Step 4:   Find the current best distance to of the points to the line.
#           Compare this to the answer found in Step 2.
#           If the difference is <= 0.005  # --> an arbitrary stopping point Dr. Kinsman picked.
#               then exit.
#           otherwise,
#               alpha <-- 0.95 * alpha     # --> alpha gets smaller
#               go back to step 2.
#           
#           Alternatively, we can stop if the increment we use to change things by gets too small.
#           In other words, when alpha gets too small.
#
#
# As this progresses, it changes the values of the parameters in a direction which decreases the 
# print -- it goes in the direction that descents the gradients of equal print.
# This is an approximation for gradient descent.
#
# Note:  The choice of alpha <-- 0.95 * alpha, uses 0.95.
#        This is an arbitrary value.
#        The 0.95 determines how fast alpha shrinks.
#        This is called THE LEARNING RATE.
#         
# In this case, we have two hyper-parameters that are used.
# 1.  The main one is the learning rate, 0.95.
# 2.  The other one is the stopping criterion, of 0.005.

    if ( LEARNING_RATE >= 1.0 ):
        print('Learning Rate cannot be greater then or equal to 1.0') 
    elif ( LEARNING_RATE <= 0 ):
        print('Learning Rate cannot be less then or equal to 0.0') 
     
    '''
    # Step 1.   Set values of theta and rho.
    theta   =   INITIAL_THETA 
    rho     =   INITIAL_RHO 
    alpha   =   INITIAL_ALPHA 
    '''
    while ( 1 ):
        #
        # Step 2.   Find the distance of all the points to the line.
        dst_step_2  = dst_from_pts_to_line( xys, theta, rho ) 
        #
        # Step 3A.  Try values of rho = rho +/- some change, alpha.
        #           If the change decreases the total distance of the points to the line,
        #           then keep making that change.
        dst_step_3a_minus   = dst_from_pts_to_line( xys, theta, rho-alpha ) 
        dst_step_3a_plus    = dst_from_pts_to_line( xys, theta, rho+alpha ) 

        if ( dst_step_3a_minus < dst_step_2 ):
            # Search in the minus  direction.
            step_for_rho        = -alpha 
            dst_best_yet        =  dst_step_3a_minus 
            rho                 =  rho + step_for_rho       # TAKE THE STEP
        elif ( dst_step_3a_plus < dst_step_2 ):
            # Search in the positive alpha direction.
            step_for_rho        = +alpha 
            dst_best_yet        =  dst_step_3a_plus 
            rho                 =  rho + step_for_rho       # TAKE THE STEP
        else:
            dst_best_yet        = dst_step_2 
            step_for_rho        = alpha                     # Has to be something.
                                                            # Do not take a step.
         

        while( 1 ):
            # Look ahead:
            dst_step_3a     = dst_from_pts_to_line( xys, theta, rho+step_for_rho ) 

            if ( dst_step_3a < dst_best_yet ):
                # Still going down, so take the step:
                rho             = rho + step_for_rho     # Rem: if step_for_rho is negative, this decraments.
                dst_best_yet    = dst_step_3a 
                
            else: 
                break 
             
         

        #
        # Step 3B.  Try values of theta = theta +/- some change, alpha.
        #           If the change decreases the total distance of the points to the line,
        #           then keep making that change.
        dst_step_3b         = dst_from_pts_to_line( xys, theta, rho ) 
        dst_step_3b_minus   = dst_from_pts_to_line( xys, theta-alpha, rho ) 
        dst_step_3b_plus    = dst_from_pts_to_line( xys, theta+alpha, rho ) 

        if ( dst_step_3b_minus < dst_step_3b ):
            # Search in the minus alpha direction.
            step_for_theta      = -alpha 
            dst_best_yet        =  dst_step_3b_minus 
            theta               =  theta + step_for_theta   # TAKE THE PREVIOUS STEP
        elif ( dst_step_3b_plus < dst_step_3b ):
            # Search in the positive alpha direction.
            step_for_theta      = +alpha 
            dst_best_yet        =  dst_step_3b_plus 
            theta               =  theta + step_for_theta   # TAKE THE PREVIOUS STEP
        else:
            dst_best_yet    = dst_step_3b 
            step_for_theta  = alpha                         # Must be something.
                                                            # Do not take a step.
                                                            #         
        while( 1 ):
            # LOOK AHEAD:
            dst_step_3b     = dst_from_pts_to_line( xys, theta+step_for_theta, rho ) 

            if ( dst_step_3b < dst_best_yet ):
                theta           = theta + step_for_theta     # Rem: if step_for_theta is negative, this decraments.
                dst_best_yet    = dst_step_3b 
                # Handle the wrap-around for degrees:
                if ( theta > 180 ):
                    theta = theta - 360 
                 
                if ( theta < -180 ):
                    theta = theta + 360 
                 
            else:
                break         

        # Flip the line from the negative angle a positive angle.
        # This negates rho, whatever that is:
        if ( theta < 0 ):
            theta = theta + 180 
            rho   = -rho 

        #
        # Step 4:   Find the current best distance to of the points to the line.
        #
        #           Compare this to the answer found in Step 2.
        #           If the difference is <= 0.005  # --> an arbitrary stopping point Dr. Kinsman picked.
        #               then exit.
        #           otherwise,
        #               alpha <-- 0.95 * alpha     # --> alpha gets smaller
        #               go back to step 2.
        #
        #
        dst_step_4  = dst_from_pts_to_line( xys, theta, rho ) 

     
        print('Rho = %+9.5f  Theta = %9.6f  ' % (rho ,  theta ) )
        print('Alpha = %+6.5f  Misclassification Rate = %8.7f\n' % (alpha , dst_step_4 )) 

        average_dist_list.append(dst_step_4)
        
        if ( dst_step_2 < dst_step_4 ):
            print('Distance is getting worse, not better.') 
         
        
        alpha = LEARNING_RATE * alpha 
        if ( alpha > MINIMUM_ALPHA_DELTA ):
            continue 
        else:
            break 

    dst_to_origin = dst_from_pts_to_line( xys, theta, rho ) 
    
    print('Answer:\n') 
    print('A                           = %+7.5f \n' %         math.cos(math.radians((theta) ))) 
    print('B                           = %+7.5f \n' %          math.sin(math.radians((theta) )))
    print('theta                       = %+7.5f degrees\n' %   theta) 
    print('rho                         = %+7.5f \n' %         rho ) 
    print('Accuray  = %+7.5f \n' %         (1- dst_to_origin) ) 

    return theta, rho
def dst_from_pts_to_line( xys, theta, rho ):
    n_pts   =  len(xys[0]) 
    A       =  math.cos(math.radians(( theta ) ))
    B       =  math.sin(math.radians(( theta ) ))
    C       =  -rho 

    ttl_dist = 0
    for cor in range(0 , n_pts):
        if((xys[0][cor]*A + xys[1][cor]*B + C )*xys[2][cor] > 0):
            ttl_dist += 1 
    return ttl_dist /n_pts 

