// Copyright (c) 2025 Pranav Jadhav. All rights reserved.
// AI Agent Orchestration Platform - React Dashboard

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { motion } from 'framer-motion';

// Layout Components
import { DashboardLayout } from './components/layout/DashboardLayout';
import { AuthLayout } from './components/layout/AuthLayout';

// Page Components
import { LoginPage } from './pages/auth/LoginPage';
import { DashboardPage } from './pages/dashboard/DashboardPage';
import { AgentsPage } from './pages/agents/AgentsPage';
import { WorkflowsPage } from './pages/workflows/WorkflowsPage';
import { ExecutionsPage } from './pages/executions/ExecutionsPage';
import { ApprovalsPage } from './pages/approvals/ApprovalsPage';
import { MonitoringPage } from './pages/monitoring/MonitoringPage';
import { SettingsPage } from './pages/settings/SettingsPage';
import { NotFoundPage } from './pages/NotFoundPage';

// Hooks and Utilities
import { useAuth } from './hooks/useAuth';
import { LoadingSpinner } from './components/ui/LoadingSpinner';

// Create Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// Protected Route Component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

// App Component
const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-background font-sans antialiased">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <Routes>
              {/* Authentication Routes */}
              <Route path="/login" element={
                <AuthLayout>
                  <LoginPage />
                </AuthLayout>
              } />

              {/* Protected Dashboard Routes */}
              <Route path="/" element={
                <ProtectedRoute>
                  <DashboardLayout />
                </ProtectedRoute>
              }>
                <Route index element={<Navigate to="/dashboard" replace />} />
                <Route path="dashboard" element={<DashboardPage />} />
                <Route path="agents" element={<AgentsPage />} />
                <Route path="workflows" element={<WorkflowsPage />} />
                <Route path="executions" element={<ExecutionsPage />} />
                <Route path="approvals" element={<ApprovalsPage />} />
                <Route path="monitoring" element={<MonitoringPage />} />
                <Route path="settings" element={<SettingsPage />} />
              </Route>

              {/* 404 Route */}
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </motion.div>

          {/* Global Toast Notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: 'hsl(var(--card))',
                color: 'hsl(var(--card-foreground))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
              },
              success: {
                style: {
                  background: 'hsl(142, 76%, 36%)',
                  color: 'white',
                },
                iconTheme: {
                  primary: 'white',
                  secondary: 'hsl(142, 76%, 36%)',
                },
              },
              error: {
                style: {
                  background: 'hsl(0, 84%, 60%)',
                  color: 'white',
                },
                iconTheme: {
                  primary: 'white',
                  secondary: 'hsl(0, 84%, 60%)',
                },
              },
            }}
          />
        </div>
      </Router>
    </QueryClientProvider>
  );
};

export default App;